# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
""" U-NET architecture."""

import numpy as np

import math

import einops.array_api

import mlx.core as mx
import mlx.nn as nn

from ml_mdm.models.unet import ResNetConfig


def _fan_in(w):
    return np.prod(w.shape[1:])


def _fan_out(w):
    return w.shape[0]


def _fan_avg(w):
    return 0.5 * (_fan_in(w) + _fan_out(w))


def init_weights(module):
    """Initialize weights of a module using PyTorch's default initialization"""
    for k, v in module.parameters().items():
        if 'weight' in k:
            if isinstance(module, nn.GroupNorm):
                # PyTorch initializes GroupNorm weights to 1
                module.parameters()[k] = mx.ones_like(v)
            else:
                # For conv and linear layers, use Kaiming uniform initialization
                fan = _fan_in(v)
                bound = 1 / np.sqrt(fan)
                module.parameters()[k] = mx.random.uniform(low=-bound, high=bound, shape=v.shape)
        elif 'bias' in k:
            module.parameters()[k] = mx.zeros_like(v)
    return module


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


class SelfAttention1D_MLX(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=8,
        num_head_channels=-1,
        use_attention_ffn=False,
        pos_emb=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj_out = zero_module_mlx(nn.Linear(channels, channels))
        if use_attention_ffn:
            self.ffn = nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, 4 * channels),
                nn.GELU(),
                zero_module_mlx(nn.Linear(4 * channels, channels)),
            )
        else:
            self.ffn = None
        if pos_emb:
            from mlx.nn import RoPE

            self.pos_emb = RoPE(dim=channels // self.num_heads)
        else:
            self.pos_emb = None

    def attention(self, q, k, v, mask=None):
        bs, length, width = q.shape
        ch = width // self.num_heads
        scale = 1 / math.sqrt(math.sqrt(ch))
        q = q.reshape(bs, length, self.num_heads, ch)
        k = k.reshape(bs, length, self.num_heads, ch)
        if self.pos_emb is not None:
            q = self.pos_emb.rotate_queries_or_keys(q.permute(0, 2, 1, 3)).permute(
                0, 2, 1, 3
            )
            k = self.pos_emb.rotate_queries_or_keys(k.permute(0, 2, 1, 3)).permute(
                0, 2, 1, 3
            )
        weight = mx.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        if mask is not None:
            mask = mask.view(mask.size(0), 1, 1, mask.size(1))
            weight = weight.masked_fill(mask == 0, float("-inf"))
        weight = mx.softmax(weight, axis=-1)
        a = mx.einsum("bhts,bshc->bthc", weight, v.reshape(bs, -1, self.num_heads, ch))
        return a.reshape(bs, length, -1)

    def forward(self, x, mask):
        # assert (self.cond_dim is not None) == (cond is not None)
        qkv = self.qkv(self.norm(x))
        q, k, v = mx.split(qkv, 3, axis=-1)
        h = self.attention(q, k, v, mask)
        h = self.proj_out(h)
        x = x + h
        if self.ffn is not None:
            x = x + self.ffn(x)
        return x


class TemporalAttentionBlock_MLX(nn.Module):
    def __init__(
        self, channels, num_heads=8, num_head_channels=-1, down=False, pos_emb=False
    ):
        super().__init__()
        self.attn = SelfAttention1D_MLX(
            channels, num_heads, num_head_channels, pos_emb=pos_emb
        )
        self.mlp = MLP_MLX(channels, multiplier=4)
        self.down = down
        if down:
            self.down_conv = nn.Conv2d(
                channels, channels, kernel_size=3, stride=2, padding=1, bias=True
            )
            self.up_conv = nn.Conv2d(
                channels, channels, kernel_size=3, stride=1, padding=1, bias=True
            )

    def forward(self, x, temb):
        x_ = x
        if self.down:
            x = einops.array_api.rearrange(x, "b c h w -> b h w c")
            x = self.down_conv(x)
            x = einops.array_api.rearrange(x, "b h w c -> b c h w")

        x = einops.array_api.rearrange(x, "b c h w -> b h w c")
        T, H, W = x.shape[0] // temb.shape[0], x.shape[2], x.shape[3]
        x = einops.array_api.rearrange(x, "(b t) h w c -> (b h w) t c", t=T)
        x = self.attn.forward(x, None)
        x = self.mlp.forward(x)
        x = einops.array_api.rearrange(x, "(b h w) t c -> (b t) h w c", h=H, w=W)        
        x = einops.array_api.rearrange(x, "b h w c -> b c h w")

        if self.down:
            x = einops.array_api.rearrange(x, "b c h w -> b h w c")
            x = nn.Upsample(scale_factor=2, mode="nearest")(x)
            x = self.up_conv(x)
            x = einops.array_api.rearrange(x, "b h w c -> b c h w")
        x = x + x_
        return x


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
        super(ResNet_MLX, self).__init__()
        self.config = config
        self.norm1 = nn.GroupNorm(
            config.num_groups_norm, 
            config.num_channels, 
            pytorch_compatible=True,
            eps=1e-5   #torch std
        )

        self.conv1 = nn.Conv2d(
            config.num_channels,
            config.output_channels,
            kernel_size=3,
            padding=1,
            bias=True
        )

        self.time_layer = nn.Linear(
            time_emb_channels,
            config.output_channels * 2  
        )

        # Initialize GroupNorm2 without special initialization
        self.norm2 = nn.GroupNorm(
            config.num_groups_norm,
            config.output_channels,
            pytorch_compatible=True,
            eps=1e-5  
        )
        self.dropout = nn.Dropout(config.dropout)

        # conv2 is zero-initialized
        self.conv2 = zero_module_mlx(
            nn.Conv2d(
                config.output_channels,
                config.output_channels,
                kernel_size=3,
                padding=1,
                bias=True
            )
        )

        # Create a 1x1 conv for the residual connection if channels don't match
        if self.config.output_channels != self.config.num_channels:
            # Rename to conv3 to match PyTorch
            self.conv3 = nn.Conv2d(
                config.num_channels,
                config.output_channels,
                kernel_size=1,
                bias=True
            )

    def forward(self, x, temb):
        print("pre norm shape: ", x.shape)
        h = self.norm1(x)
        print("post norm shape: ", h.shape)
        h = nn.silu(h)
        h = self.conv1(h)

        temb_out = nn.silu(temb)
        temb_out = self.time_layer(temb_out)
        temb_out = mx.expand_dims(mx.expand_dims(temb_out, axis=1), axis=1)
        ta, tb = mx.split(temb_out, 2, axis=-1)

        # Handle batch size mismatch
        if h.shape[0] > ta.shape[0]:
            N = h.shape[0] // ta.shape[0]
            ta = mx.repeat(ta, N, axis=0)
            tb = mx.repeat(tb, N, axis=0)

        # Broadcast temporal embeddings
        ta = mx.broadcast_to(ta, h.shape)
        tb = mx.broadcast_to(tb, h.shape)

        h = nn.silu(self.norm2(h) * (1 + ta) + tb)
        h = self.dropout(h)
        h = self.conv2(h)

        # Handle residual connection
        if self.config.output_channels != self.config.num_channels:
            x = self.conv3(x)

        return h + x

