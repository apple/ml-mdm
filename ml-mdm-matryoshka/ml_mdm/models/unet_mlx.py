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
            # Reshape mask to match attention shape
            # From [bs, seq_len] to [bs * num_heads, 1, seq_len]
            expanded_mask = einops.array_api.repeat(
                mask[:, None, :],  # Add dimension for broadcasting
                "b 1 s -> (b h) 1 s",
                h=self.num_heads,
            )
            # Apply mask
            weight = mx.where(expanded_mask, weight, float("-inf"))

        weight = mx.softmax(weight, axis=-1)

        return mx.einsum(
            "bts,bcs->bct", weight, v.reshape(bs * self.num_heads, ch, -1)
        ).reshape(bs, width, length)

    def forward(self, x, cond=None, cond_mask=None):

        x = einops.array_api.rearrange(x, "b c h w -> b h w c")
        b, h, w, c = x.shape
        
        qkv = self.qkv(self.norm(x))
        qkv = einops.array_api.rearrange(qkv, "b h w (three c) -> three b (h w) c", three=3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn_output = self.attention(q, k, v)
        
        if self.cond_dim is not None and cond is not None:
            kv_cond = self.kv_cond(self.norm_cond(cond))
            kv_cond = einops.array_api.rearrange(kv_cond, "b s (two c) -> two b s c", two=2)
            k_cond, v_cond = kv_cond[0], kv_cond[1]
            attn_cond = self.attention(q, k_cond, v_cond, cond_mask)
            attn_output += attn_cond
            
        attn_output = einops.array_api.rearrange(attn_output, "b (h w) c -> b h w c", h=h, w=w)
        h = self.proj_out(attn_output)
        
        x = einops.array_api.rearrange(x, "b h w c -> b c h w")
        h = einops.array_api.rearrange(h, "b h w c -> b c h w") 
        x = x + h
        
        if self.ffn is not None:
            x = einops.array_api.rearrange(x, "b c h w -> b h w c")
            x = self.ffn(x) + x
            x = einops.array_api.rearrange(x, "b h w c -> b c h w")
            
        return x

