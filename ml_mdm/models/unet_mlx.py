# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.

import einops

import mlx.core as mx
import mlx.nn as nn

from ml_mdm.models.unet import ResNetConfig


def zero_module_mlx(module):
    """
    Zero out the parameters of an MLX module and return it.
    """
    # Create a new parameter dictionary with all parameters replaced by zeros
    zeroed_params = {
        name: mx.zeros(param.shape, dtype=param.dtype)
        for name, param in module.parameters().items()
    }
    # Update the module's parameters with the zeroed parameters
    module.update(zeroed_params)
    return module


class MLP_MLX(nn.Module):  # mlx based nn.Module
    def __init__(self, channels, multiplier=4):
        super().__init__()
        ### use mlx layers
        self.main = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, multiplier * channels),
            nn.GELU(),
            zero_module_mlx(nn.Linear(multiplier * channels, channels)),
        )

    def forward(self, x):
        return x + self.main(x)


class ResNet_MLX(nn.Module):
    def __init__(self, time_emb_channels, config: ResNetConfig):
        # TODO(ndjaitly): What about scales of weights.
        super(ResNet_MLX, self).__init__()
        self.config = config
        self.norm1 = nn.GroupNorm(config.num_groups_norm, config.num_channels)
        self.conv1 = nn.Conv2d(
            config.num_channels,
            config.output_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.time_layer = nn.Linear(time_emb_channels, config.output_channels * 2)
        self.norm2 = nn.GroupNorm(config.num_groups_norm, config.output_channels)
        self.dropout = nn.Dropout(config.dropout)
        self.conv2 = zero_module_mlx(
            nn.Conv2d(
                config.output_channels,
                config.output_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            )
        )
        if self.config.output_channels != self.config.num_channels:
            self.conv3 = nn.Conv2d(
                config.num_channels, config.output_channels, kernel_size=1, bias=True
            )

    def forward(self, x, temb):
        h = nn.silu(self.norm1(x))
        h = self.conv1(h)
        ta, tb = (
            self.time_layer(nn.silu(temb)).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        )
        if h.size(0) > ta.size(0):  # HACK. repeat to match the shape.
            N = h.size(0) // ta.size(0)
            ta = einops.repeat(ta, "b c h w -> (b n) c h w", n=N)
            tb = einops.repeat(tb, "b c h w -> (b n) c h w", n=N)
        h = nn.silu(self.norm2(h) * (1 + ta) + tb)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.config.output_channels != self.config.num_channels:
            x = self.conv3(x)
        return h + x

    def __call__(self, x, temb):
        return self.forward(x, temb)
