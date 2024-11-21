# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
""" Nested U-NET architecture."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from torchinfo import summary

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ml_mdm import config, s3_helpers
from ml_mdm.models.unet import UNet, UNetConfig, zero_module


@config.register_model_config("nested_unet", "nested_unet")
@dataclass
class NestedUNetConfig(UNetConfig):
    inner_config: UNetConfig = field(
        default_factory=lambda: UNetConfig(nesting=True),
        metadata={"help": "inner unet used as middle blocks"},
    )
    skip_mid_blocks: bool = field(default=True)
    skip_cond_emb: bool = field(default=True)
    skip_inner_unet_input: bool = field(
        default=False,
        metadata={
            "help": "If enabled, the inner unet only received the downsampled image, no features."
        },
    )
    skip_normalization: bool = field(
        default=False,
    )
    initialize_inner_with_pretrained: str = field(
        default=None,
        metadata={
            "help": (
                "Initialize the inner unet with pretrained vision model ",
                "Provide the vision_model_path",
            )
        },
    )
    freeze_inner_unet: bool = field(default=False)
    interp_conditioning: bool = field(
        default=False,
    )


@config.register_model_config("nested2_unet", "nested_unet")
@dataclass
class Nested2UNetConfig(NestedUNetConfig):
    inner_config: NestedUNetConfig = field(
        default_factory=lambda: NestedUNetConfig(nesting=True, initialize_inner_with_pretrained=None)
    )


@config.register_model_config("nested3_unet", "nested_unet")
@dataclass
class Nested3UNetConfig(Nested2UNetConfig):
    inner_config: Nested2UNetConfig = field(
        default_factory=lambda: Nested2UNetConfig(nesting=True, initialize_inner_with_pretrained=None)
    )


@config.register_model_config("nested4_unet", "nested_unet")
@dataclass
class Nested4UNetConfig(Nested3UNetConfig):
    inner_config: Nested3UNetConfig = field(
        default_factory=lambda: Nested3UNetConfig(nesting=True, initialize_inner_with_pretrained=None)
    )


def download(vision_model_path):
    import os

    from distributed import get_local_rank

    local_file = vision_model_path.replace("/", "_")
    if get_local_rank() == 0 and (not os.path.exists(local_file)):
        try:
            s3_helpers.download_object_from_full_path(
                vision_model_path, download_path=local_file
            )
        except Exception:
            pass
    if dist.is_initialized():
        dist.barrier()
    return local_file


@config.register_model("nested_unet")
class NestedUNet(UNet):
    def __init__(self, input_channels, output_channels, config: NestedUNetConfig):
        super().__init__(input_channels, output_channels=output_channels, config=config)
        config.inner_config.conditioning_feature_dim = config.conditioning_feature_dim
        if getattr(config.inner_config, "inner_config", None) is None:
            self.inner_unet = UNet(input_channels, output_channels, config.inner_config)
        else:
            self.inner_unet = NestedUNet(
                input_channels, output_channels, config.inner_config
            )

        if not config.skip_inner_unet_input:
            self.in_adapter = zero_module(
                nn.Conv2d(
                    config.resolution_channels[-1],
                    config.inner_config.resolution_channels[0],
                    kernel_size=3,
                    padding=1,
                    bias=True,
                )
            )
        else:
            self.in_adapter = None
        self.out_adapter = zero_module(
            nn.Conv2d(
                config.inner_config.resolution_channels[0],
                config.resolution_channels[-1],
                kernel_size=3,
                padding=1,
                bias=True,
            )
        )

        self.is_temporal = [config.temporal_mode and (not config.temporal_spatial_ds)]
        if hasattr(self.inner_unet, "is_temporal"):
            self.is_temporal += self.inner_unet.is_temporal

        nest_ratio = int(2 ** (len(config.resolution_channels) - 1))
        if self.is_temporal[0]:
            nest_ratio = int(np.sqrt(nest_ratio))
        if (
            self.inner_unet.config.nesting
            and self.inner_unet.model_type == "nested_unet"
        ):
            self.nest_ratio = [
                nest_ratio * self.inner_unet.nest_ratio[0]
            ] + self.inner_unet.nest_ratio
        else:
            self.nest_ratio = [nest_ratio]

        if config.initialize_inner_with_pretrained is not None:
            try:
                self.inner_unet.load(download(config.initialize_inner_with_pretrained))
            except Exception as e:
                print("<-- load pretrained checkpoint error -->")
                print(f"{e}")

        if config.freeze_inner_unet:
            for p in self.inner_unet.parameters():
                p.requires_grad = False
        if config.interp_conditioning:
            self.interp_layer1 = nn.Linear(self.temporal_dim // 4, self.temporal_dim)
            self.interp_layer2 = nn.Linear(self.temporal_dim, self.temporal_dim)

    @property
    def model_type(self):
        return "nested_unet"

    def forward_conditioning(self, *args, **kwargs):
        return self.inner_unet.forward_conditioning(*args, **kwargs)

    def forward_denoising(
        self, x_t, times, cond_emb=None, conditioning=None, cond_mask=None, micros={}
    ):
        # 1. time embedding
        temb = self.create_temporal_embedding(times)
        if cond_emb is not None:
            temb = temb + cond_emb
        if self.conditions is not None:
            temb = temb + self.forward_micro_conditioning(times, micros)

        # 2. input layer (normalize the input)
        if self._config.nesting:
            x_t, x_feat = x_t
        bsz = [x.size(0) for x in x_t]
        bh, bl = bsz[0], bsz[1]
        x_t_low, x_t = x_t[1:], x_t[0]
        x = self.forward_input_layer(
            x_t, normalize=(not self.config.skip_normalization)
        )
        if self._config.nesting:
            x = x + x_feat

        # 3. downsample blocks in the outer layers
        x, skip_activations = self.forward_downsample(
            x,
            temb[:bh],
            conditioning[:bh],
            cond_mask[:bh] if cond_mask is not None else cond_mask,
        )

        # 4. run inner unet
        x_inner = self.in_adapter(x) if self.in_adapter is not None else None
        x_inner = (
            torch.cat([x_inner, x_inner.new_zeros(bl - bh, *x_inner.size()[1:])], 0)
            if bh < bl
            else x_inner
        )  # pad zeros for low-resolutions
        x_low, x_inner = self.inner_unet.forward_denoising(
            (x_t_low, x_inner), times, cond_emb, conditioning, cond_mask, micros
        )
        x_inner = self.out_adapter(x_inner)
        x = x + x_inner[:bh] if bh < bl else x + x_inner

        # 5. upsample blocks in the outer layers
        x = self.forward_upsample(
            x,
            temb[:bh],
            conditioning[:bh],
            cond_mask[:bh] if cond_mask is not None else cond_mask,
            skip_activations,
        )

        # 6. output layer
        x_out = self.forward_output_layer(x)

        # 7. outpupt both low and high-res output
        if isinstance(x_low, list):
            out = [x_out] + x_low
        else:
            out = [x_out, x_low]
        if self._config.nesting:
            return out, x
        return out

    def print_size(self, target_image_size=256):
        pass
        # ratio = self.nest_ratio
        # summary(self, [
        #     (1, self.input_channels, target_image_size, target_image_size),  # x_t
        #     (1, self.input_channels, target_image_size // ratio, target_image_size // ratio),  # x_t_low
        #     (1,),  # times
        #     (1, 32, self.input_conditioning_feature_dim),  # conditioning
        #     (1, 32)],   # condition_mask
        #     dtypes=[torch.float, torch.float, torch.float, torch.float],
        #     col_names=["input_size", "output_size", "num_params"],
        #     row_settings=["var_names"],
        #     depth=4)
