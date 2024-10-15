# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
import logging
import math
from dataclasses import dataclass, field
from enum import Enum

from einops import repeat
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Type(Enum):
    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return str(self)


class ScheduleType(Type):
    COSINE = 0
    DDPM = 2
    DEEPFLOYD = 3

    @staticmethod
    def argparse(s):
        try:
            return ScheduleType[s.upper()]
        except KeyError:
            return s


class PredictionType(Type):
    DDPM = 3
    DDIM = 4
    V_PREDICTION = 5

    @staticmethod
    def argparse(s):
        try:
            return PredictionType[s.upper()]
        except KeyError:
            return s


class ThresholdType(Type):
    NONE = 0
    CLIP = 1
    DYNAMIC = 2
    DYNAMIC_IF = 3

    @staticmethod
    def argparse(s):
        try:
            return ThresholdType[s.upper()]
        except KeyError:
            return s


@dataclass
class SamplerConfig:
    num_diffusion_steps: int = field(
        default=32, metadata={"help": "Number of diffusion steps"}
    )
    reproject_signal: bool = field(
        default=False,
        metadata={"help": "Whether to reproject signal back to noise level"},
    )
    schedule_type: ScheduleType = field(
        default=ScheduleType.DDPM, metadata={"help": "Type of schedule"}
    )
    prediction_type: PredictionType = field(
        default=PredictionType.DDPM,
        metadata={"help": ("Type of target (DDPM, DDIM, V_PREDICTION)")},
    )
    loss_target_type: PredictionType = field(
        default=None, metadata={"help": ("Type of target (DDPM, DDIM, V_PREDICTION)")}
    )
    beta_start: float = field(
        default=0.0001, metadata={"help": "Start beta for HA schedule"}
    )
    beta_end: float = field(default=0.02, metadata={"help": "End beta for HA schedule"})
    threshold_function: ThresholdType = field(
        default=ThresholdType.CLIP,
        metadata={"help": "thresholding function used for clipping x0 values"},
    )
    rescale_schedule: float = field(
        default=1.0,
        metadata={
            "help": "rescale the standard noise scheduler, useful for high-resolution images."
        },
    )
    rescale_signal: float = field(
        default=None,
        metadata={
            "help": (
                "instead of directly rescaling the noise schedule weights, only rescale the image, "
                "see details: https://arxiv.org/pdf/2301.10972.pdf"
            )
        },
    )
    schedule_shifted: bool = field(
        default=False,
        metadata={
            "help": "automatically shift the noise schedule based on the resolution ratios."
        },
    )
    schedule_shifted_power: float = field(
        default=1,
        metadata={
            "help": "noise shifted ratio, by default using 1."
        },
    )


##########################################################################################
#                                    Noise schedule                                      #
##########################################################################################


def schedule_cosine(
    timesteps: int, logsnr_min: float = -5.0, logsnr_max: float = 5.0
) -> np.ndarray:
    """See DDPMs distillation paper. https://arxiv.org/abs/2202.00512"""
    t = np.linspace(0.0, 1.0, num=timesteps)
    b = np.arctan(np.exp(-0.5 * logsnr_max))
    a = np.arctan(np.exp(-0.5 * logsnr_min)) - b
    logsnrs = -2.0 * np.log(np.tan(a * t + b))
    gammas = 1 / (1 + np.exp(-logsnrs))
    gammas = np.concatenate(([1.0], gammas))
    return gammas


def schedule_ddpm_defults(
    timesteps: int, beta_start: float, beta_end: float
) -> np.ndarray:
    """https://arxiv.org/pdf/2006.11239.pdf"""
    betas = np.concatenate(([0], np.linspace(beta_start, beta_end, num=timesteps)))
    log_alphas = np.log(1.0 - betas)
    gammas = np.exp(np.cumsum(log_alphas))
    return gammas


def squaredcos_cap_v2(timesteps: int):
    """
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py#L147
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = [0]
    for i in range(timesteps):
        t1 = i / timesteps
        t2 = (i + 1) / timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
    betas = np.asarray(betas)
    log_alphas = np.log(1.0 - betas)
    gammas = np.exp(np.cumsum(log_alphas))
    return gammas


##########################################################################################
#                              Sampler Class                                             #
##########################################################################################


class Sampler(nn.Module):
    def __init__(self, sampler_config: SamplerConfig):
        super().__init__()
        self.n_steps = sampler_config.num_diffusion_steps

        # original noise schedule based on defined family
        self.get_noise_schedule(
            sampler_config.schedule_type,
            sampler_config.num_diffusion_steps,
            sampler_config,
        )

        logging.info(f"Step gammas: {self.gammas}")
        self._config = sampler_config

        # by default, we use the same loss for the model predictions
        if self._config.loss_target_type is None:
            self._config.loss_target_type = self._config.prediction_type

    def read_gamma(self, time, image):
        B, C, H, W = image.size()
        time = repeat(time, "b -> b c h w", c=C, h=H, w=W)
        return self.gammas[time]

    def get_noise_schedule(self, schedule_type, n_steps, sampler_config):
        # pre-defined noise schedule functions
        if schedule_type == ScheduleType.COSINE:
            _gammas = schedule_cosine(n_steps)
        elif schedule_type == ScheduleType.DDPM:
            _gammas = schedule_ddpm_defults(
                n_steps, sampler_config.beta_start, sampler_config.beta_end
            )
        elif schedule_type == ScheduleType.DEEPFLOYD:
            _gammas = squaredcos_cap_v2(n_steps)
        else:
            raise Exception("Unknown")

        self.register_buffer(name="_gammas", tensor=torch.tensor(_gammas).float())

        # rescale noise schedule
        scale_factor = sampler_config.rescale_schedule
        gammas = self._gammas.clone()
        gammas = self.get_schedule_shifted(gammas, scale_factor)

        # VDM loss weighting
        g = gammas[2:]
        g_last = gammas[1:-1]
        # weights = (g_last - g)/ (1-g_last) / (1-g)
        weights = g_last * (1 - g) / (1 - g_last) / g - 1
        weights = torch.cat([weights[:1], weights[:1], weights])

        self.register_buffer("gammas", gammas)
        self.register_buffer("vdm_loss_weights", weights)

    def get_eps_time(self, images, time=None):
        batch_size = images.shape[0]
        if time is None:
            time = torch.randint(0, self.n_steps, (batch_size,), device=images.device)
        else:
            time = time * torch.ones(batch_size, dtype=torch.long, device=images.device)
        g, g_last = self.read_gamma(time + 1, images), self.read_gamma(time, images)
        weights = self.vdm_loss_weights[time + 1]
        eps = torch.randn_like(images)
        return eps, g, g_last, weights, time

    def get_xt(self, images, eps, g):
        x_t = g.sqrt() * images + (1 - g).sqrt() * eps
        return x_t

    def get_image_rescaled(self, images, scale_factor=None):
        if scale_factor is None:
            scale_factor = self._config.rescale_signal
        if scale_factor:  # divide the signal
            images = images / scale_factor
        return images

    def get_schedule_shifted(self, gammas, scale_factor=None):
        if (scale_factor is not None) and (scale_factor > 1):  # rescale noise schecule
            p = self._config.schedule_shifted_power
            scale_factor = scale_factor ** p
            snr = gammas / (1 - gammas)
            scaled_snr = snr / scale_factor
            gammas = 1 / (1 + 1 / scaled_snr)
        return gammas

    def get_prediction_targets(self, images, eps, g, g_last, prediction_type=None):
        if prediction_type is None:
            prediction_type = self._config.loss_target_type

        if (prediction_type == PredictionType.DDPM) or (
            prediction_type == PredictionType.DDIM
        ):
            pred = eps
        elif prediction_type == PredictionType.V_PREDICTION:
            # pred = (x_t * g.sqrt() - images) / (1-g).sqrt()
            pred = g.sqrt() * eps - (1 - g).sqrt() * images
        else:
            raise Exception("Unsupported type")
        return pred

    def get_prediction_xt_last(
        self,
        x_t,
        pred,
        g,
        g_last,
        prediction_type=None,
        clip_fn=None,
        need_noise=False,
        ddim_eta=None,
        input_noise=None,
        image_scale=None,
    ):
        """
        x_t:             noisy image
        pred:            model prediction (can be x0, eps, v, etc)
        g:               noise level at t
        g_last:          noise level at t_last
        prediction_type: model prediction type (can be x0, eps, v, etc)
        clip_fn:         clipping function, by default clip(-1, 1)
        need_noise:      use noise or not
        ddim_eta:        if None, then not using DDIM, otherwise, use DDIM implementation (1==DDPM)
        """

        if prediction_type is None:
            prediction_type = self._config.prediction_type

        # 1. get beta, beta_tilde
        alpha = g / g_last
        beta = 1 - alpha
        beta_tilde = beta * (1 - g_last) / (1 - g)

        x0 = self.get_x0_eps_from_pred(
            x_t, pred, g, prediction_type=prediction_type, return_eps=False
        )
        # 2. clipping
        image_scale = 1 if image_scale is None else image_scale
        x0 = (
            torch.clip(x0, -image_scale, image_scale) / image_scale
            if clip_fn is None
            else clip_fn(x0, image_scale)
        )

        # 3.common DDPM/DDIM implementation
        if ddim_eta is None:
            x_t_last = x0 * beta * g_last.sqrt() / (1 - g) + x_t * alpha.sqrt() * (
                1 - g_last
            ) / (1 - g)

        else:
            eps = (x_t - x0 * g.sqrt()) / (1 - g).sqrt()
            if ddim_eta > 0:
                beta_tilde = (ddim_eta**2) * beta_tilde
                x_t_last = x0 * g_last.sqrt() + eps * (1 - g_last - beta_tilde).sqrt()
            else:
                need_noise = False
                x_t_last = x0 * g_last.sqrt() + eps * (1 - g_last).sqrt()

        if need_noise:  # need noise
            if input_noise is None:
                input_noise = torch.randn_like(x_t_last)
            x_t_last = x_t_last + beta_tilde.sqrt() * input_noise

        # 4. get the equvilent noise
        eps = (x_t_last - g_last.sqrt() * x0) / (1 - g_last).sqrt()
        return x0, x_t_last, eps

    def get_x0_eps_from_pred(
        self, x_t, pred, g, prediction_type=None, clip_fn=None, return_eps=True
    ):
        batch_size = x_t.size(0)
        if prediction_type is None:
            prediction_type = self._config.prediction_type

        if (prediction_type == PredictionType.DDPM) or (
            prediction_type == PredictionType.DDIM
        ):
            x0 = (x_t - pred * (1 - g).sqrt()) / g.sqrt()
        elif prediction_type == PredictionType.V_PREDICTION:
            x0 = x_t * g.sqrt() - pred * (1 - g).sqrt()
        else:
            raise Exception("prediction type not set to a correct value")

        if clip_fn is not None:
            x0 = clip_fn(x0)

        if not return_eps:
            return x0
        eps = (x_t - x0 * g.sqrt()) / (1 - g).sqrt()
        return x0, eps

    def get_pred_from_x0_xt(self, x_t, x0, g, prediction_type=None):
        batch_size = x_t.size(0)
        if prediction_type is None:
            prediction_type = self._config.prediction_type

        if (prediction_type == PredictionType.DDPM) or (
            prediction_type == PredictionType.DDIM
        ):
            pred = (x_t - x0 * g.sqrt()) / (1 - g).sqrt()
        elif prediction_type == PredictionType.V_PREDICTION:
            pred = (g.sqrt() * x_t - x0) / (1 - g).sqrt()
        else:
            raise Exception("prediction type not set to a correct value")
        return pred

    def get_xt_minus_1(
        self,
        model,
        time_step,
        x_t,
        lm_outputs,
        lm_mask,
        micros={},
        time_step_last=None,
        guidance_scale=1,
        ddim_eta=None,
        return_details=False,
    ):
        batch_size = x_t.shape[0]
        ones = torch.ones(batch_size, dtype=torch.long, device=self.gammas.device)
        if time_step_last is None:
            time_step_last = time_step - 1
        t = ones * time_step
        s = ones * time_step_last

        # NOTE: be careful t-1 to the model input. variance is not used for now.
        g, g_last = self.read_gamma(t, x_t), self.read_gamma(s, x_t)
        pred, _ = self.forward_model(
            model, x_t, t - 1, lm_outputs, lm_mask, micros, guidance_scale
        )
        x0, x_s, _ = self.get_prediction_xt_last(
            x_t,
            pred,
            g,
            g_last,
            prediction_type=self._config.prediction_type,
            need_noise=(time_step_last != 0),
            ddim_eta=ddim_eta,
            clip_fn=self.clip_sample,
            image_scale=self._config.rescale_signal,
        )

        if return_details:
            extra = (g, g_last)
            return x0, x_s, extra
        else:
            return x_s

    def forward_model(
        self, model, x_t, t, lm_outputs, lm_mask, micros={}, guidance_scale=1
    ):
        if guidance_scale != 1:
            assert x_t.shape[0] * 2 == lm_outputs.shape[0]
            pred, extras = model(
                torch.cat([x_t] * 2),
                torch.cat([t, t]),
                lm_outputs,
                lm_mask,
                micros=micros,
            )
            pred_uncond, pred = pred.chunk(2)
            pred = pred_uncond + guidance_scale * (pred - pred_uncond)
            extras = extras.chunk(2)[1]
        else:
            pred, extras = model(x_t, t, lm_outputs, lm_mask, micros)
        return pred, extras

    def _threshold_sample(
        self, sample, dynamic_thresholding_ratio=0.995, sample_max_value=100
    ):
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."

        https://arxiv.org/abs/2205.11487
        """
        dtype = sample.dtype
        batch_size, channels, height, width = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = (
                sample.float()
            )  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * height * width)
        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(abs_sample, dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(s, min=1, max=sample_max_value)
        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = (
            torch.clamp(sample, -s, s) / s
        )  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, height, width)
        sample = sample.to(dtype)

        return sample

    def clip_sample(self, pred_x0, image_scale=1):
        s = image_scale
        if self._config.threshold_function == ThresholdType.CLIP:
            return (pred_x0 * s).clip(-1, 1) / s
        elif self._config.threshold_function == ThresholdType.DYNAMIC:
            return self._threshold_sample(pred_x0 * s, 0.995, 100) / s
        elif self._config.threshold_function == ThresholdType.DYNAMIC_IF:
            return self._threshold_sample(pred_x0 * s, 0.95, 1.5) / s
        return pred_x0

    def sample(self, *args, **kwargs):
        if not kwargs.get("yield_output", False):
            output = self._sample(*args, **kwargs)
            return next(output)
        return self._sample(*args, **kwargs)

    def _sample(
        self,
        model,
        x_t,
        lm_outputs,
        lm_mask,
        micros,
        return_sequence=False,
        use_beta_tilde=False,
        t=-1,
        num_inference_steps=2000,
        ddim_eta=None,
        guidance_scale=1,
        resample_steps=False,
        disable_bar=True,
        yield_output=False,
        **post_args,
    ):
        """
        Starting with x_t, at time step t, perform diffusion to first step.
        """
        assert not (yield_output and return_sequence), "not allowed."

        if not resample_steps:
            # this follows the original Navdeep implementation, using full steps
            num_inference_steps = self.n_steps

        timesteps = self.set_timesteps(num_inference_steps)
        timesteps = torch.from_numpy(timesteps).to(x_t.device)
        if t > -1:  # filter steps larger than t
            timesteps = timesteps[timesteps <= t]

        seq = []
        if return_sequence:
            seq.append(x_t)

        for i, t in tqdm(
            enumerate(timesteps[:-1]), total=num_inference_steps, disable=disable_bar
        ):
            t_last = timesteps[i + 1] if resample_steps else None
            x0, x_t, extra = self.get_xt_minus_1(
                model,
                t,
                x_t,
                lm_outputs,
                lm_mask,
                micros,
                time_step_last=t_last,
                guidance_scale=guidance_scale,
                ddim_eta=ddim_eta,
                return_details=True,
            )
            if yield_output:
                yield self._postprocess(x_t, x0, extra, **post_args)

            if return_sequence:
                seq.append(self._postprocess(x_t))

        if return_sequence:
            seq[-1] = torch.clip(seq[-1], -1, 1)
            yield seq
        else:
            yield self._postprocess(x_t, x0, extra, clip=True, **post_args)

    def _postprocess(
        self,
        x_t,
        x0=None,
        extra=None,
        yield_full=False,
        clip=False,
        image_scale=None,
        **unused,
    ):
        if image_scale is None:
            image_scale = self._config.rescale_signal
        if image_scale:
            x0 = x0 * image_scale
            x_t = x_t * image_scale
        if clip:
            x_t = torch.clip(x_t, -1, 1)
        if yield_full:
            return (x0, x_t, extra)
        return x_t

    def set_timesteps(self, num_inference_steps=250):
        step_ratio = (self._config.num_diffusion_steps + 1) / (num_inference_steps + 1)
        timesteps = (
            (np.arange(0, num_inference_steps + 1) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        return timesteps


class NestedSampler(Sampler):
    def get_gammas(self, gamma, scales, images=None):
        if not self._config.schedule_shifted:
            gammas = [gamma for _ in scales]
        else:
            gammas = [self.get_schedule_shifted(gamma, s) for s in scales]
        if images is not None and gammas[0].size(-1) != 1:
            gammas = [
                F.interpolate(g, im.size(-1), mode="nearest")
                for g, im in zip(gammas, images)
            ]
        return gammas

    def get_xt(self, x0, eps, g, scales):
        x_t = []
        for x, s, e, gi in zip(x0, scales, eps, g):
            x_t += [
                super().get_xt(
                    self.get_image_rescaled(x, s)
                    if not self._config.schedule_shifted
                    else x,
                    e,
                    gi,
                )
            ]
        return x_t

    def get_prediction_targets(self, x0, eps, g, g_last, scales, prediction_type=None):
        tgt = []
        for x, s, e, gi, gil in zip(x0, scales, eps, g, g_last):
            tgt += [
                super().get_prediction_targets(
                    self.get_image_rescaled(x, s)
                    if not self._config.schedule_shifted
                    else x,
                    e,
                    gi,
                    gil,
                    prediction_type,
                )
            ]
        return tgt

    def get_xt_minus_1(
        self,
        model,
        time_step,
        x_t,
        lm_outputs,
        lm_mask,
        micros={},
        time_step_last=None,
        guidance_scale=1,
        ddim_eta=None,
        return_details=False,
    ):
        scales = model.vision_model.nest_ratio + [1]
        if isinstance(x_t, torch.Tensor):
            out = [x_t]
            for s in scales[1:]:
                ratio = scales[0] // s
                x_t_low = F.avg_pool2d(x_t, ratio) * ratio
                x_t_low = x_t_low.normal_()
                out += [x_t_low]
            x_t = out

        batch_size = x_t[0].shape[0]
        ones = torch.ones(batch_size, dtype=torch.long, device=self.gammas.device)
        t = ones * time_step
        s = t - 1 if time_step_last is None else ones * time_step_last

        # NOTE: be careful t-1 to the model input. variance is not used for now.
        g_t = self.get_gammas(self.read_gamma(t, x_t[0]), scales, x_t)
        g_s = self.get_gammas(self.read_gamma(s, x_t[0]), scales, x_t)
        p_t = self.forward_model(
            model, x_t, t - 1, lm_outputs, lm_mask, micros, guidance_scale
        )
        x0, x_s, eps = map(
            list,
            zip(
                *[
                    self.get_prediction_xt_last(
                        x,
                        p,
                        g,
                        g_last,
                        prediction_type=self._config.prediction_type,
                        need_noise=time_step != 1,
                        ddim_eta=ddim_eta,
                        clip_fn=self.clip_sample,
                        image_scale=s if not self._config.schedule_shifted else 1,
                    )
                    for x, p, g, g_last, s in zip(x_t, p_t, g_t, g_s, scales)
                ]
            ),
        )

        if return_details:
            extra = (g_t[-1], g_s[-1])
            return x0, x_s, extra
        else:
            return x_s

    def _postprocess(
        self,
        x_t,
        x0=None,
        extra=None,
        yield_full=False,
        clip=False,
        output_inner=False,
        **unused,
    ):
        scales = [
            x_t[i].size(-1) / x_t[-1].size(-1)
            if not self._config.schedule_shifted
            else 1
            for i in range(len(x_t))
        ]
        out = super()._postprocess(
            x_t[0],
            x0[0] if x0 is not None else x0,
            extra,
            yield_full=yield_full,
            clip=clip,
            image_scale=scales[0],
            **unused,
        )

        def cat(x, size):
            # nx = x.new_ones(x.size(0), 3, size, size)
            # nx[..., :x.size(-2), :x.size(-1)] = x
            nx = F.interpolate(x, size, mode="bilinear")
            return nx

        # output inner loop results.
        if output_inner:
            outs = [out]
            for i in range(1, len(x_t)):
                outs += [
                    super()._postprocess(
                        x_t[i],
                        x0[i] if x0 is not None else x0[i],
                        extra,
                        yield_full=yield_full,
                        clip=clip,
                        image_scale=scales[i],
                        **unused,
                    )
                ]

            if not yield_full:
                out = torch.cat(
                    [cat(oi, size=outs[0].size(-1)) for oi in outs[::-1]], -1
                )
            else:
                x0, xt, extra = zip(*outs)
                x0 = torch.cat([cat(xi, size=x0[0].size(-1)) for xi in x0[::-1]], -1)
                xt = torch.cat([cat(xi, size=xt[0].size(-1)) for xi in xt[::-1]], -1)
                out = (x0, xt, extra[-1])
        return out

    def forward_model(
        self, model, x_t, t, lm_outputs, lm_mask, micros={}, guidance_scale=1
    ):
        def cfg(pred):
            pred_uncond, pred = pred.chunk(2)
            return pred_uncond + guidance_scale * (pred - pred_uncond)

        if guidance_scale != 1:
            assert x_t[0].shape[0] * 2 == lm_outputs.shape[0]
            p_t = model(
                [torch.cat([x] * 2) for x in x_t],
                torch.cat([t] * 2),
                lm_outputs,
                lm_mask,
                micros,
            )
            p_t = [cfg(p) for p in p_t]
        else:
            p_t = model(x_t, t, lm_outputs, lm_mask, micros)
        return p_t
