# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
""" Basic UNet-DDPM pipeline. """
import logging
from dataclasses import dataclass, field
from typing import List

from einops import rearrange

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from ml_mdm import config, samplers


def sv(x, f):
    save_image(x, f, value_range=(-1, 1), normalize=True)


##########################################################################################
#                     Standard UNet Diffusion                                            #
##########################################################################################


@config.register_pipeline_config("unet")
@dataclass
class DiffusionConfig:
    sampler_config: samplers.SamplerConfig = field(
        default=samplers.SamplerConfig(), metadata={"help": "Sampler configuration"}
    )
    model_output_scale: float = field(
        default=0,
        metadata={
            "help": "If non-zero, predictions are scaled +/- scale value with tanh"
        },
    )
    use_vdm_loss_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Use Variational Diffusion Model loss "
                "(https://openreview.net/forum?id=2LdBqxc1Yv)"
            )
        },
    )


class Model(nn.Module):
    def __init__(
        self, vision_model, diffusion_config: DiffusionConfig = DiffusionConfig()
    ):
        super().__init__()
        self.diffusion_config = diffusion_config
        self._output_scale = diffusion_config.model_output_scale
        self.vision_model = vision_model
        self.sampler = None

    def set_sampler(self, sampler):
        self.sampler = sampler

    def load(self, vision_file):
        return self.vision_model.load(vision_file)

    def save(self, vision_file, other_items=None):
        self.vision_model.save(vision_file, other_items=other_items)

    @property
    def input_channels(self):
        return self.vision_model.input_channels

    def forward(
        self,
        x_t: torch.Tensor,
        times: torch.Tensor,
        lm_outputs: torch.Tensor,
        lm_mask: torch.Tensor,
        micros: {},
    ) -> (torch.Tensor, torch.Tensor):
        outputs = self.vision_model(x_t, times, lm_outputs, lm_mask, micros)
        if self._output_scale != 0:
            outputs = torch.tanh(outputs / self._output_scale) * self._output_scale
        return outputs, torch.ones_like(outputs)


@config.register_pipeline("unet")
class Diffusion(nn.Module):
    def __init__(self, denoising_model, diffusion_config: DiffusionConfig):
        super().__init__()
        logging.info(f"Diffusion config: {diffusion_config}")
        self.model = Model(denoising_model, diffusion_config)
        self.sampler = samplers.Sampler(diffusion_config.sampler_config)
        self.model.set_sampler(self.sampler)

        self._config = diffusion_config
        self.loss_fn = nn.MSELoss(reduction="none")

    def get_model(self):
        if hasattr(self.model, "module"):
            return self.model.module
        return self.model

    def to(self, device):
        self.model = self.model.to(device)
        self.sampler = self.sampler.to(device)
        return self

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
        self.sampler.eval()

    def get_xt_minus_1(self, t, x_t, lm_outputs, lm_mask):
        self.eval()
        return self.sampler.get_xt_minus_1(t, x_t, lm_outputs, lm_mask)

    def get_pred_for_training(self, x_t, pred, g):
        if (
            self._config.sampler_config.loss_target_type
            == self._config.sampler_config.prediction_type
        ):
            return pred
        else:
            x0, _ = self.sampler.get_x0_eps_from_pred(
                x_t, pred, g, self._config.sampler_config.prediction_type
            )
            pred = self.sampler.get_pred_from_x0_xt(
                x_t, x0, g, self._config.sampler_config.loss_target_type
            )
            return pred

    def get_micro_conditioning(self, sample):
        micros, conditions = {}, self.get_model().vision_model.conditions
        if conditions is not None:
            micros = {key: sample[key] for key in conditions if key in sample}
        return micros

    def get_loss(self, sample):
        images, lm_outputs, lm_mask = (
            sample["images"],
            sample["lm_outputs"],
            sample["lm_mask"],
        )

        # 1. get the parameters
        eps, g, g_last, weights, time = self.sampler.get_eps_time(images)
        if not self._config.use_vdm_loss_weights:
            weights = None

        x_t = self.sampler.get_xt(self.sampler.get_image_rescaled(images), eps, g)
        micros = self.get_micro_conditioning(sample)

        # 2. get model predictions
        means, variances = self.model(x_t, time, lm_outputs, lm_mask, micros)

        # 3. compute loss
        tgt = self.sampler.get_prediction_targets(
            images, eps, g, g_last, self._config.sampler_config.loss_target_type
        )
        pred = self.get_pred_for_training(x_t, means, g)
        loss = self.loss_fn(pred, tgt).mean(axis=(1, 2, 3))
        return loss, time, x_t, means, tgt, weights

    def get_noise(self, num_examples, input_channels, image_side, device):
        return torch.randn(num_examples, input_channels, image_side, image_side).to(
            device
        )

    def sample(self, num_examples, sample, image_side, device, **kwargs):
        self.eval()
        noise = self.get_noise(
            num_examples, self.get_model().input_channels, image_side, device
        )
        lm_outputs, lm_mask = sample["lm_outputs"], sample["lm_mask"]
        micros = self.get_micro_conditioning(sample)
        return self.sampler.sample(
            self.get_model(), noise, lm_outputs, lm_mask, micros, **kwargs
        )

    def partial_diffusion(
        self, images, t, lm_outputs, lm_mask, device, return_sequence=False
    ):
        self.eval()
        (_, x_t, _, _) = self.sampler.get_noisy_samples_for_training(images, t)
        return self.sampler.sample(
            x_t, lm_outputs, lm_mask, return_sequence=return_sequence, t=t
        )


##########################################################################################
#                     Nested-UNet Diffusion                                              #
##########################################################################################


@config.register_pipeline_config("nested_unet")
@dataclass
class NestedDiffusionConfig(DiffusionConfig):
    use_double_loss: bool = field(
        default=False,
        metadata={"help": "only useful for nested unet, loss on two resolutions."},
    )
    multi_res_weights: str = field(
        default=None,
        metadata={
            "help": "if not None, setting additional weights for different resolutions, like 4:1"
        },
    )
    no_use_residual: bool = field(
        default=False, metadata={"help": ("Do not use residual for 256x256 modules ")}
    )
    use_random_interp: bool = field(
        default=False,
        metadata={
            "help": "randomly apply downsample interpolation for high-resolution during training."
        },
    )
    mixed_ratio: str = field(
        default=None,
        metadata={"help": "batch size ratio for different resolutions, e.g., 0.5"},
    )
    random_downsample: bool = field(
        default=False, metadata={"help": "only useful for temporal mode"}
    )
    average_downsample: bool = field(
        default=False,
    )
    mid_downsample: bool = field(
        default=False,
    )


class NestedModel(Model):
    def forward(
        self,
        x_t: List[torch.Tensor],
        times: torch.Tensor,
        lm_outputs: torch.Tensor,
        lm_mask: torch.Tensor,
        micros={},
        mixed_ratio: List[float] = None,
    ) -> (torch.Tensor, torch.Tensor):
        batch_size = x_t[0].size(0)
        if mixed_ratio is not None:  # skip computation for part of the high-resolution
            x_t = [x[: int(m * x.size(0))] for x, m in zip(x_t, mixed_ratio)]

        # call the vision model
        p_t = self.vision_model(x_t, times, lm_outputs, lm_mask, micros)

        if (
            mixed_ratio is not None
        ):  # pad the output to zero for this part of computations
            p_t = [
                torch.cat([p, p.new_zeros(batch_size - p.size(0), *p.size()[1:])], 0)
                for p in p_t
            ]

        # recompute the noise from pred_low
        if not self.diffusion_config.no_use_residual:
            assert (
                self.diffusion_config.mixed_ratio is None
            ), "do not support mixed-batch"
            x_t, x_t_low = x_t
            pred, pred_low = p_t
            pred_x0_low, _ = self.sampler.get_x0_eps_from_pred(x_t_low, pred_low, times)
            pred_x0_low = pred_x0_low.clamp(
                min=-1, max=1
            )  # by force, clip the x0 values.
            pred_x0_low = (
                F.interpolate(pred_x0_low, scale_factor=ratio, mode="bicubic") / ratio
            )
            pred = pred + self.sampler.get_pred_from_x0_xt(x_t, pred_x0_low, times)
            p_t = [pred, pred_low]
        return p_t


@config.register_pipeline("nested_unet")
class NestedDiffusion(Diffusion):
    def __init__(self, denoising_model, diffusion_config: DiffusionConfig):
        super(Diffusion, self).__init__()

        logging.info(f"Diffusion config: {diffusion_config}")
        self.model = NestedModel(denoising_model, diffusion_config)
        self.sampler = samplers.NestedSampler(diffusion_config.sampler_config)
        self.model.set_sampler(self.sampler)

        self._config = diffusion_config
        self.loss_fn = nn.MSELoss(reduction="none")

        self.mixed_ratio = None
        if self._config.mixed_ratio:
            self.mixed_ratio = np.cumsum(
                np.asarray([float(x) for x in self._config.mixed_ratio.split(":")])
            )
            self.mixed_ratio = self.mixed_ratio / self.mixed_ratio[-1]

    def get_loss(self, sample):
        images, lm_outputs, lm_mask = (
            sample["images"],
            sample["lm_outputs"],
            sample["lm_mask"],
        )
        micros = self.get_micro_conditioning(sample)

        # 1. get the parameters
        scales = self.get_model().vision_model.nest_ratio + [1]
        ratios = [scales[0] // s for s in scales]
        istime = [False] + self.get_model().vision_model.is_temporal

        eps, g, g_last, weights, time = self.sampler.get_eps_time(images)
        if not self._config.use_vdm_loss_weights:
            weights = None

        # 2. get the low-resolution images
        _images, _eps, T = [images], [eps], 4
        for iz, (r, ist) in enumerate(zip(ratios, istime)):
            if iz == 0:
                continue
            prev_r = ratios[iz - 1]
            rr = r // prev_r
            x = _images[-1]
            if ist:
                x = rearrange(x, "b c (n h) (m w) -> b (n m) c h w", n=T, m=T)
                x = x[:, :: (r * r)]
                T = T // rr
                x = rearrange(x, "b (n m) c h w -> b c (n h) (m w)", n=T, m=T)
            else:
                x = F.avg_pool2d(x, rr)
            _images += [x]
            _eps += [F.avg_pool2d(_eps[-1], rr) * rr]
        images, eps = _images, _eps

        # rescale noise schedule
        g = self.sampler.get_gammas(g, scales, images)
        g_last = self.sampler.get_gammas(g_last, scales, images)

        for i in range(1, len(eps)):
            eps[i] = eps[i].normal_()  # make the noise random

        # 3. get model predictions (high, low resolutions, by default. only 2 so far)
        x_t = self.sampler.get_xt(images, eps, g, scales)
        p_t = self.model(x_t, time, lm_outputs, lm_mask, micros, self.mixed_ratio)

        # 4. prepare for the targets and compute loss
        tgt = self.sampler.get_prediction_targets(
            images, eps, g, g_last, scales, self._config.sampler_config.loss_target_type
        )
        pred = [self.get_pred_for_training(x, p, gi) for x, p, gi in zip(x_t, p_t, g)]
        loss = 0
        if self._config.multi_res_weights is not None:
            assert (
                self._config.use_double_loss
            ), "only makes sense when applying more losses"
            w = [float(w) for w in self._config.multi_res_weights.split(":")]
        else:
            w = [1.0] * len(x_t)
        for i in range(len(x_t)):
            if i == 0 or self._config.use_double_loss:
                loss_ = self.loss_fn(pred[i], tgt[i]).mean(axis=(1, 2, 3))
                if self.mixed_ratio is not None:
                    loss_ = loss_ / self.mixed_ratio[i]
                    loss_[
                        int(self.mixed_ratio[i] * loss_.size(0)) :
                    ] = 0  # discard these losses
            else:
                loss_ = pred[i].mean() * 0.0
            loss_ = loss_ * w[i]
            loss = loss + loss_

        return loss, time, x_t[0], pred[0], tgt[0], weights
