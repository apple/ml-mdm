# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.

import math

import einops.array_api

import mlx.core as mx
import mlx.nn as nn


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


class SelfAttention_MLX(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=8,
        num_head_channels=-1,
        cond_dim=None,
        use_attention_ffn=False,
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
        self.norm = nn.GroupNorm(32, channels, pytorch_compatible=True)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.cond_dim = cond_dim
        if cond_dim is not None and cond_dim > 0:
            self.norm_cond = nn.LayerNorm(cond_dim)
            self.kv_cond = nn.Linear(cond_dim, channels * 2)
        self.proj_out = zero_module_mlx(nn.Conv2d(channels, channels, 1))
        if use_attention_ffn:
            self.ffn = nn.Sequential(
                nn.GroupNorm(32, channels, pytorch_compatible=True),
                nn.Conv2d(channels, 4 * channels, 1),
                nn.GELU(),
                zero_module_mlx(nn.Conv2d(4 * channels, channels, 1)),
            )
        else:
            self.ffn = None

    def attention(self, q, k, v, mask=None):
        bs, width, length = q.shape
        ch = width // self.num_heads
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = mx.einsum(
            "bct,bcs->bts",
            (q * scale).reshape(bs * self.num_heads, ch, length),
            (k * scale).reshape(bs * self.num_heads, ch, -1),
        )  # More stable with f16 than dividing afterwards
        if mask is not None:
            mask = (
                mask.view(mask.size(0), 1, 1, mask.size(1))
                .repeat(1, self.num_heads, 1, 1)
                .flatten(0, 1)
            )
            weight = weight.masked_fill(mask == 0, float("-inf"))
        weight = mx.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = mx.einsum("bts,bcs->bct", weight, v.reshape(bs * self.num_heads, ch, -1))
        return a.reshape(bs, -1, length)

    def forward(self, x, cond=None, cond_mask=None):
        # assert (self.cond_dim is not None) == (cond is not None)
        b, c, *spatial = x.shape
        # x = x.reshape(b, c, -1)
        x = einops.array_api.rearrange(x, "b c h w -> b h w c")
        print("rearrranged input tensor: ", x.shape)
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(b, 3 * c, -1).chunk(3, dim=1)
        h = self.attention(q, k, v)
        if self.cond_dim is not None and self.cond_dim > 0:
            kv_cond = self.kv_cond(self.norm_cond(cond)).transpose(-2, -1)
            k_cond, v_cond = kv_cond.chunk(2, dim=1)
            h_cond = self.attention(q, k_cond, v_cond, cond_mask)
            h = h + h_cond
        h = h.reshape(b, c, *spatial)
        h = self.proj_out(h)
        x = x + h
        if self.ffn is not None:
            x = self.ffn(x) + x
        return x
