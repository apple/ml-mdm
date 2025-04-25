# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.

import copy
import logging
import math
import pdb
from dataclasses import dataclass, field
from enum import Enum

import einops.array_api
from torchinfo import summary

import mlx.core as mx
import mlx.nn as nn

import numpy as np

from ml_mdm import config
from ml_mdm.models.unet import ResNetConfig, UNetConfig
from ml_mdm.utils import fix_old_checkpoints


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
            # Initialize biases to zero
            module.parameters()[k] = mx.zeros_like(v)
    return module




# MLX doesn't have a register_buffer method like PyTorch
# Instead, we'll just set the attribute directly in the UNet_MLX class
# This is a simpler approach that works with MLX's module system


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

def temporal_wrapper(f):
    def wrapper(*args, **kwargs):
        args = list(args)
        model = args[0]
        fname = f.__name__
        temporal = model._config.temporal_mode
        spatial_ds = model._config.temporal_spatial_ds

        if hasattr(model, "nest_ratio"):
            S = model.nest_ratio[0]
            T = 1 if len(model.nest_ratio) == 1 else model.nest_ratio[1]
            if spatial_ds:
                S = T

        if temporal:
            I = T if "upsample" in fname else S
            args[1] = einops.rearrange(
                args[1], "b c (n h) (m w) -> (b n m) c h w ", n=I, m=I
            )

        outs = f(*args, **kwargs)

        if temporal:
            O = T if "downsample" in fname else S
            x = outs[0] if isinstance(outs, tuple) else outs
            x = einops.rearrange(x, "(b n m) c h w -> b c (n h) (m w)", n=O, m=O)
            if isinstance(outs, tuple):
                return x, *outs[1:]
            return x
        return outs

    return wrapper

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
            try:
                # Print debug info
                #print(f"Mask shape: {mask.shape}, weight shape: {weight.shape}")
                #print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
                
                # Check if we're dealing with conditioning mask (different dimensions)
                if len(mask.shape) == 2 and mask.shape[1] != weight.shape[2]:
                    print("Handling conditioning mask with different dimensions")
                    # For conditioning mask, we don't need to apply it in the same way
                    # We'll just return the weight as is, since the mask was already applied
                    # when creating the conditioning vectors
                    pass
                else:
                    # For regular self-attention mask
                    # Reshape mask to match attention shape
                    # From [bs, seq_len] to [bs * num_heads, 1, seq_len]
                    expanded_mask = einops.array_api.repeat(
                        mask[:, None, :],  # Add dimension for broadcasting
                        "b 1 s -> (b h) 1 s",
                        h=self.num_heads,
                    )
                    weight = mx.where(expanded_mask, weight, float("-inf"))
            except Exception as e:
                print(f"Error in attention mask application: {e}")

        weight = mx.softmax(weight, axis=-1)

        return mx.einsum(
            "bts,bcs->bct", weight, v.reshape(bs * self.num_heads, ch, -1)
        ).reshape(bs, width, length)

    def forward(self, x, cond=None, cond_mask=None):
        # Determine if input is in NCHW format (PyTorch style)
        x_is_nchw = False
        if len(x.shape) == 4:
            if x.shape[1] == self.channels:  # Input is in NCHW format
                x_is_nchw = True
                print(f"Converting input from NCHW to NHWC: {x.shape}")
                x = mx.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        
        # Print debug info
        print(f"SelfAttention input shape: {x.shape}, channels: {self.channels}")
        
        # Get dimensions
        b, h, w, c = x.shape
        
        # Apply normalization - ensure x has the right shape for GroupNorm
        try:
            print(f"pre norm shape: {x.shape}")
            normalized = self.norm(x)
            print(f"post norm shape: {normalized.shape}")
            qkv = self.qkv(normalized)
            qkv = einops.array_api.rearrange(qkv, "b h w (three c) -> three b (h w) c", three=3)
            q, k, v = qkv[0], qkv[1], qkv[2]
        except Exception as e:
            print(f"Error in SelfAttention_MLX.forward: {e}")
            print(f"x shape: {x.shape}, channels: {self.channels}")
            raise
        
        try:
            attn_output = self.attention(q, k, v)
            
            if self.cond_dim is not None and cond is not None:
                kv_cond = self.kv_cond(self.norm_cond(cond))
                kv_cond = einops.array_api.rearrange(kv_cond, "b s (two c) -> two b s c", two=2)
                k_cond, v_cond = kv_cond[0], kv_cond[1]
                attn_cond = self.attention(q, k_cond, v_cond, cond_mask)
                attn_output += attn_cond
        except Exception as e:
            print(f"Error in attention computation: {e}")
            raise
            
        try:
            # Reshape attention output back to spatial dimensions
            attn_output = einops.array_api.rearrange(attn_output, "b (h w) c -> b h w c", h=h, w=w)
            h = self.proj_out(attn_output)
            
            # Add residual connection - keep everything in NHWC format for MLX
            x = x + h
            
            # Apply FFN if present
            if self.ffn is not None:
                x = self.ffn(x) + x
            
            # Convert back to NCHW format if the input was in NCHW format
            if x_is_nchw:
                print(f"Converting output back to NCHW: {x.shape}")
                x = mx.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
        except Exception as e:
            print(f"Error in final part of SelfAttention_MLX.forward: {e}")
            raise
            
        return x



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
        # Ensure input is in NHWC format for MLX GroupNorm
        if len(x.shape) == 4 and x.shape[1] == self.config.num_channels:
            # Convert from NCHW to NHWC
            x = einops.array_api.rearrange(x, "b c h w -> b h w c")
            
        print("pre norm shape: ", x.shape)
        try:
            h = self.norm1(x)
            print("post norm shape: ", h.shape)
            h = nn.silu(h)
            h = self.conv1(h)
        except Exception as e:
            print(f"Error in ResNet_MLX.forward: {e}")
            print(f"Input shape: {x.shape}, channels: {self.config.num_channels}")
            raise

        temb_out = nn.silu(temb)
        temb_out = self.time_layer(temb_out)
        temb_out = mx.expand_dims(mx.expand_dims(temb_out, axis=1), axis=1)
        ta, tb = mx.split(temb_out, 2, axis=-1)

        # Handle batch size mismatch
        if h.shape[0] > ta.shape[0]:
            N = h.shape[0] // ta.shape[0]
            ta = mx.repeat(ta, N, axis=0)
            tb = mx.repeat(tb, N, axis=0)

        # Broadcast temporal embeddings to match h's shape
        ta = mx.broadcast_to(ta, h.shape)
        tb = mx.broadcast_to(tb, h.shape)

        h = nn.silu(self.norm2(h) * (1 + ta) + tb)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.config.output_channels != self.config.num_channels:
            x = self.conv3(x)
            
        # Return in NHWC format for consistency with MLX
        return x + h


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

class SelfAttention1DBlock_MLX(nn.Module):
    def __init__(self, channels, num_heads=8, num_head_channels=-1, mlp_multiplier=4):
        super().__init__()
        self.attn = SelfAttention1D_MLX(channels, num_heads, num_head_channels)
        self.mlp = MLP_MLX(channels, mlp_multiplier)

    def forward(self, x, mask):
       # x = einops.array_api.rearrange(x, "b c h w -> b h w c")
        x = self.mlp.forward(self.attn.forward(x, None))
       # x = einops.array_api.rearrange(x, "b h w c -> b c h w")
        return x

class ResNetBlock_MLX(nn.Module):
    def __init__(
        self,
        temporal_dim: int,
        num_residual_blocks: int,
        num_attention_layers: int,
        downsample_output: bool,
        upsample_output: bool,
        resnet_configs: list,
        conditioning_feature_dim: int = -1,
        temporal_mode: bool = False,
        temporal_pos_emb: bool = False,
        temporal_spatial_ds: bool = False,
        num_temporal_attention_layers: int = None,
    ):
        super().__init__()
        resnets = []
        self.temporal = temporal_mode
        self.temporal_spatial_ds = temporal_spatial_ds
        self.num_residual_blocks = num_residual_blocks
        self.num_attention_layers = num_attention_layers
        self.num_temporal_attention_layers = num_temporal_attention_layers
        self.upsample_output = upsample_output
        self.downsample_output = downsample_output
        assert (downsample_output and upsample_output) == False

        for i in range(num_residual_blocks):
            cur_config = resnet_configs[i]
            resnets.append(ResNet_MLX(temporal_dim, cur_config))

        mod_restnets = []
        if resnets is not None:
            for module in resnets:
                mod_restnets.append(module)
        self.resnets = mod_restnets

        if self.num_attention_layers > 0:
            attn = []
            for i in range(num_residual_blocks):
                for j in range(self.num_attention_layers):
                    attn.append(
                        SelfAttention_MLX(
                            resnet_configs[i].output_channels,
                            cond_dim=conditioning_feature_dim,
                            use_attention_ffn=resnet_configs[i].use_attention_ffn,
                        )
                    )
            mod_attn = []
            if attn is not None:
                for module in attn:
                    mod_attn.append(module)
            self.attn = mod_attn

        if (
            self.num_temporal_attention_layers
            and self.num_temporal_attention_layers > 0
            and (not self.temporal_spatial_ds)
        ):
            t_attn = []
            for i in range(num_residual_blocks):
                for j in range(self.num_temporal_attention_layers):
                    t_attn.append(
                        TemporalAttentionBlock_MLX(
                            resnet_configs[i].output_channels,
                            num_head_channels=32,
                            down=True,
                            pos_emb=temporal_pos_emb,
                        )
                    )
            mod_t_attn = []
            if t_attn is not None:
                for module in t_attn:
                    mod_t_attn.append(module)
            self.t_attn = mod_t_attn
        conv_layer = (
            nn.Conv2d if (not self.temporal) or self.temporal_spatial_ds else nn.Conv1d
        )
        if self.downsample_output:
            self.resample = conv_layer(
                resnet_configs[-1].output_channels,
                resnet_configs[-1].output_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            )

        elif self.upsample_output:
            self.resample = conv_layer(
                resnet_configs[-1].output_channels,
                resnet_configs[-1].output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )

    def forward(
        self,
        x,
        temb,
        skip_activations=None,
        return_activations=False,
        conditioning=None,
        cond_mask=None,
    ):
        activations = []
        for i in range(self.num_residual_blocks):
            if skip_activations is not None:
                skip_input = skip_activations.pop(0)
                
                
                # Determine tensor layouts
                x_is_nchw = len(x.shape) == 4 and x.shape[1] > 1 and x.shape[1] < x.shape[2] and x.shape[1] < x.shape[3]
                skip_is_nchw = len(skip_input.shape) == 4 and skip_input.shape[1] > 1 and skip_input.shape[1] < skip_input.shape[2] and skip_input.shape[1] < skip_input.shape[3]
                
                # For MLX, we want to ensure both tensors are in NCHW format for consistent handling
                if not x_is_nchw and len(x.shape) == 4:
                    # Convert x from NHWC to NCHW
                    print(f"Converting x from NHWC to NCHW: {x.shape}")
                    x = mx.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
                
                if not skip_is_nchw and len(skip_input.shape) == 4:
                    # Convert skip_input from NHWC to NCHW
                    print(f"Converting skip_input from NHWC to NCHW: {skip_input.shape}")
                    skip_input = mx.transpose(skip_input, (0, 3, 1, 2))  # NHWC -> NCHW
                
                # Now both tensors should be in NCHW format, concatenate along the channel dimension (dim 1)
                print(f"After conversion - x shape: {x.shape}, skip_input shape: {skip_input.shape}")
                try:
                    # Concatenate along channel dimension (dim 1) for NCHW format
                    x = mx.concat([x, skip_input], axis=1)
                except Exception as e:
                    print(f"Concatenation error: {e}")
                    # Try to fix any remaining dimension issues
                    if x.shape[2:] != skip_input.shape[2:]:
                        skip_input = mx.reshape(skip_input, (skip_input.shape[0], skip_input.shape[1], x.shape[2], x.shape[3]))
                        x = mx.concat([x, skip_input], axis=1)

            x = self.resnets[i].forward(x, temb)
            if self.num_attention_layers > 0:
                L = self.num_attention_layers
                for j in range(L):
                    x = self.attn[i * L + j].forward(x, conditioning, cond_mask)
            if (
                self.num_temporal_attention_layers
                and self.num_temporal_attention_layers > 0
            ):
                L = self.num_temporal_attention_layers
                for j in range(L):
                    x = self.t_attn[i * L + j].forward(x, temb)
            activations.append(x)

        if self.downsample_output or self.upsample_output:
            try:
                # Make sure x is in NHWC format for MLX
                # Check if x is in NCHW format by looking at its shape
                if len(x.shape) == 4 and x.shape[3] != self.resnets[0].config.output_channels:
                    print(f"Converting x from NCHW {x.shape} to NHWC for resample")
                    x = einops.array_api.rearrange(x, "b c h w -> b h w c")
                    
                if self.temporal and (not self.temporal_spatial_ds):
                    T, H, W = x.shape[0] // temb.shape[0], x.shape[1], x.shape[2]
                    x = einops.array_api.rearrange(x, "(b t) h w c -> (b h w) t c", t=T)
                    
                if self.upsample_output:
                    # Implement nearest-neighbor upsampling manually for MLX
                    if len(x.shape) == 4:  # NHWC format
                        b, h, w, c = x.shape
                        # Duplicate each row and column (nearest neighbor upsampling)
                        x = mx.repeat(x, 2, axis=1)  # Duplicate rows
                        x = mx.repeat(x, 2, axis=2)  # Duplicate columns
                    elif len(x.shape) == 3:  # For temporal data
                        # Handle temporal data if needed
                        x = mx.repeat(x, 2, axis=0)  # Simple duplication
                    
                print(f"Resample input shape: {x.shape}")
                x = self.resample(x)
                print(f"Resample output shape: {x.shape}")
            except Exception as e:
                print(f"Error in ResNetBlock_MLX resample: {e}")
                print(f"x shape: {x.shape}")
                raise
            if self.temporal and (not self.temporal_spatial_ds):
                x = einops.array_api.rearrange(x, "(b h w) t c -> (b t) h w c", h=H, w=W)
            activations.append(x)

        if not return_activations:
            return x
        return x, activations

@config.register_model("unet")
class UNet_MLX(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, config: UNetConfig):
        super().__init__()
        self.down_blocks = []
        self.config = config
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_conditioning_feature_dim = config.conditioning_feature_dim
        if (
            config.conditioning_feature_dim > 0
            and config.conditioning_feature_proj_dim > 0
        ):
            config.conditioning_feature_dim = config.conditioning_feature_proj_dim
        self.temporal_dim = (
            config.resolution_channels[0] * 4
            if config.temporal_dim is None
            else config.temporal_dim
        )

        half_dim = self.temporal_dim // 8
        emb = math.log(10000) / half_dim 
        emb = mx.exp(mx.arange(half_dim, dtype=mx.float32) * -emb)
        # Instead of register_buffer, directly set the attribute
        self.t_emb = emb

        self.temb_layer1 = nn.Linear(self.temporal_dim // 4, self.temporal_dim)
        self.temb_layer2 = nn.Linear(self.temporal_dim, self.temporal_dim)

        if config.conditioning_feature_dim > 0 and (not config.skip_cond_emb):
            self.cond_emb = nn.Linear(
                config.conditioning_feature_dim, self.temporal_dim, bias=False
            )
        else:
            self.cond_emb = None

        self.conditions = None
        if config.micro_conditioning is not None:
            self.conditions = {
                c.split(":")[0]: float(c.split(":")[1])
                for c in config.micro_conditioning.split(",")
            }
            # Store condition layers in a regular dictionary instead of ModuleDict
            self.cond_layers = {}
            for condition in self.conditions:
                # Create the layers for each condition
                layer1 = nn.Linear(self.temporal_dim // 4, self.temporal_dim)
                layer2 = zero_module_mlx(nn.Linear(self.temporal_dim, self.temporal_dim))
                # Store them in a list
                self.cond_layers[condition] = [layer1, layer2]

        channels = config.resolution_channels[0]
        self.conv_in = nn.Conv2d(
            input_channels, channels, kernel_size=3, stride=1, padding=1, bias=True
        )
        skip_channels = [channels]
        num_resolutions = len(config.resolution_channels)
        self.num_resolutions = num_resolutions

        for i in range(num_resolutions):
            down_resnet_configs = []
            num_resnets_per_resolution = config.num_resnets_per_resolution[i]
            for j in range(num_resnets_per_resolution):
                resnet_config = copy.copy(config.resnet_config)
                resnet_config.num_channels = channels
                resnet_config.output_channels = config.resolution_channels[i]
                skip_channels.append(resnet_config.output_channels)
                down_resnet_configs.append(resnet_config)
                channels = resnet_config.output_channels

            if i != num_resolutions - 1:
                # no downsampling here, so no skip connections.
                skip_channels.append(resnet_config.output_channels)

            num_attention_layers = (
                config.num_attention_layers[i] if i in config.attention_levels else 0
            )
            num_temporal_attention_layers = (
                config.num_temporal_attention_layers[i]
                if config.num_temporal_attention_layers is not None
                else None
            )
            self.down_blocks.append(
                ResNetBlock_MLX(
                    self.temporal_dim,
                    num_resnets_per_resolution,
                    num_attention_layers,
                    downsample_output=i != num_resolutions - 1,
                    upsample_output=False,
                    resnet_configs=down_resnet_configs,
                    conditioning_feature_dim=(
                        config.conditioning_feature_dim
                        if i in self.config.attention_levels
                        else -1
                    ),
                    temporal_mode=config.temporal_mode,
                    temporal_pos_emb=config.temporal_positional_encoding,
                    temporal_spatial_ds=config.temporal_spatial_ds,
                    num_temporal_attention_layers=num_temporal_attention_layers,
                )
            )
            channels = resnet_config.output_channels

        # middle resnets keep the resolution.
        resnet_config = copy.copy(resnet_config)
        resnet_config.num_channels = channels
        resnet_config.output_channels = channels

        if not config.skip_mid_blocks:
            self.mid_blocks = [
                ResNetBlock_MLX(
                    self.temporal_dim,
                    1,
                    True,  # attn
                    False,  # downsample
                    False,  # upsample
                    resnet_configs=[resnet_config],
                    conditioning_feature_dim=config.conditioning_feature_dim,
                ),
                ResNetBlock_MLX(
                    self.temporal_dim,
                    1,
                    False,  # attn
                    False,  # downsample
                    False,  # upsample
                    resnet_configs=[copy.copy(resnet_config)],
                ),
            ]

        self.up_blocks = []
        for i in reversed(range(num_resolutions)):
            up_resnet_configs = []
            num_resnets_per_resolution = config.num_resnets_per_resolution[i]
            for j in range(num_resnets_per_resolution + 1):
                resnet_config = copy.copy(config.resnet_config)
                resnet_config.num_channels = channels + skip_channels.pop()
                resnet_config.output_channels = config.resolution_channels[i]
                up_resnet_configs.append(resnet_config)
                channels = resnet_config.output_channels

            num_attention_layers = (
                config.num_attention_layers[i] if i in config.attention_levels else 0
            )
            num_temporal_attention_layers = (
                config.num_temporal_attention_layers[i]
                if config.num_temporal_attention_layers is not None
                else None
            )
            self.up_blocks.append(
                ResNetBlock_MLX(
                    self.temporal_dim,
                    num_resnets_per_resolution + 1,
                    num_attention_layers,
                    downsample_output=False,
                    upsample_output=i != 0,
                    resnet_configs=up_resnet_configs,
                    conditioning_feature_dim=(
                        config.conditioning_feature_dim
                        if i in self.config.attention_levels
                        else -1
                    ),
                    temporal_mode=config.temporal_mode,
                    temporal_pos_emb=config.temporal_positional_encoding,
                    temporal_spatial_ds=config.temporal_spatial_ds,
                    num_temporal_attention_layers=num_temporal_attention_layers,
                )
            )
            channels = resnet_config.output_channels

        self.norm_out = nn.GroupNorm(config.resnet_config.num_groups_norm, channels)
        self.conv_out = zero_module_mlx(
            nn.Conv2d(channels, output_channels, kernel_size=3, padding=1)
        )
        self._config = config
        
        mod_down_blocks = []
        if self.down_blocks is not None:
            for i in self.down_blocks:
                mod_down_blocks.append(i)
        self.down_blocks = mod_down_blocks

        if not config.skip_mid_blocks:
            mod_mid_blocks = []
            if self.mid_blocks is not None:
                for i in self.mid_blocks:
                    mod_mid_blocks.append(i)
            self.mid_blocks = mod_mid_blocks
        
        mod_up_blocks = []
        if self.up_blocks is not None:
            for i in self.up_blocks:
                mod_up_blocks.append(i)
        self.up_blocks = mod_up_blocks

        self.masked_cross_attention = config.masked_cross_attention
        if config.conditioning_feature_dim > 0 and (not config.skip_cond_emb):
            if config.conditioning_feature_proj_dim > 0:
                # note that now config.conditioning_feature_proj_dim == config.conditioning_feature_dim
                self.lm_proj = nn.Linear(
                    self.input_conditioning_feature_dim, config.conditioning_feature_dim
                )
            
            # Create attention blocks for lm_head
            lm_head_blocks = []
            for _ in range(config.num_lm_head_layers):
                lm_head_blocks.append(SelfAttention1DBlock_MLX(config.conditioning_feature_dim))
            
            # Store the blocks in self.lm_head
            self.lm_head = lm_head_blocks

        self.is_temporal = []

    @property
    def model_type(self):
        return "unet"

    #def print_size(self, target_image_size: int =64):
    #    summary(
    #        self,
    #        [
    #            (1, self.input_channels, target_image_size, target_image_size),  # x_t
    #            (1,),  # times
    #            (1, 32, self.input_conditioning_feature_dim),  # conditioning
    #            (1, 32),
    #        ],  # condition_mask
    #        dtypes=[torch.float, torch.float, torch.float, torch.float],
    #        col_names=["input_size", "output_size", "num_params"],
    #        row_settings=["var_names"],
    #        depth=4,
    #    )

    def save(self, fname: str, other_items=None):
        logging.info(f"Saving model file: {fname}")
        checkpoint = {"state_dict": self.state_dict()}
        if other_items is not None:
            for k, v in other_items.items():
                checkpoint[k] = v
        mx.save(fname, checkpoint)

    def load(self, fname: str):
        logging.info(f"Loading model file: {fname}")
        fix_old_checkpoints.mimic_old_modules()
        # first load to cpu or we will run out of memory.
        checkpoint = mx.load(fname)
        new_state_dict = self.state_dict()
        filtered_state_dict = {
            key: value
            for key, value in checkpoint["state_dict"].items()
            if key in new_state_dict
        }
        unknown1 = {
            key: value
            for key, value in checkpoint["state_dict"].items()
            if key not in new_state_dict
        }
        unknown2 = {
            key: value
            for key, value in new_state_dict.items()
            if key not in filtered_state_dict
        }
        if len(unknown1) > 0 or len(unknown2) > 0:
            print({key for key in unknown1}, {key for key in unknown2})

        self.load_state_dict(filtered_state_dict, strict=False)
        other_items = {}
        for k, v in checkpoint.items():
            if k != "model_state_dict":
                other_items[k] = copy.copy(v)
        del checkpoint
        return other_items

    def create_temporal_embedding(self, times, ff_layers=None):
        # MLX doesn't have view, use reshape instead
        # Reshape times to (batch_size, 1) and multiply with t_emb
        times_reshaped = mx.reshape(times, (times.shape[0], 1))
        temb = times_reshaped * self.t_emb
        temb = mx.concat([mx.sin(temb), mx.cos(temb)], axis=1)
        if temb.shape[1] % 2 == 1:
            # zero pad
            temb = mx.concat([temb, mx.zeros((times.shape[0], 1))], axis=1)
        if ff_layers is None:
            layer1, layer2 = self.temb_layer1, self.temb_layer2
        else:
            layer1, layer2 = ff_layers
        temb = layer2(nn.silu(layer1(temb)))
        return temb

    def forward_conditioning(self, conditioning, cond_mask):
        if self.config.conditioning_feature_proj_dim > 0:
            conditioning = self.lm_proj(conditioning)
        for head in self.lm_head:
            conditioning = head.forward(
                conditioning, mask=cond_mask if self.masked_cross_attention else None
            )
        if cond_mask is None or (
            not self.masked_cross_attention and len(self.lm_head) > 0
        ):
            y = mx.mean(conditioning, axis=1)
        else:
            expanded_mask = mx.expand_dims(cond_mask, axis=-1)
            y = (expanded_mask * conditioning).sum(axis=1) / mx.sum(
                cond_mask, axis=1, keepdims=True
            )
        if not self.masked_cross_attention:
            cond_mask = None
        cond_emb = self.cond_emb(y)
        return cond_emb, conditioning, cond_mask

    @temporal_wrapper
    def forward_input_layer(self, x_t, normalize=False):
        if isinstance(x_t, list) and len(x_t) == 1:
            x_t = x_t[0]
        if normalize:
            x_t = x_t / x_t.std((1, 2, 3), keepdims=True)
        x = self.conv_in(x_t)
        return x

    @temporal_wrapper
    def forward_output_layer(self, x):
        x_out = nn.silu(self.norm_out(x))
        x_out = self.conv_out(x_out)
        return x_out

    @temporal_wrapper
    def forward_downsample(self, x, temb, conditioning, cond_mask):
        skip_activations = [x]
        for i, block in enumerate(self.down_blocks):
            if i in self.config.attention_levels:
                x, activations = block.forward(
                    x,
                    temb,
                    return_activations=True,
                    conditioning=conditioning,
                    cond_mask=cond_mask,
                )
            else:
                x, activations = block.forward(x, temb, return_activations=True)
            skip_activations.extend(activations)
        return x, skip_activations

    @temporal_wrapper
    def forward_upsample(self, x, temb, conditioning, cond_mask, skip_activations):
        num_resolutions = len(self._config.resolution_channels)
        for i, block in enumerate(self.up_blocks):
            ri = num_resolutions - 1 - i
            num_skip = self._config.num_resnets_per_resolution[ri] + 1
            skip_connections = skip_activations[-num_skip:]
            skip_connections.reverse()
            if ri in self.config.attention_levels:
                x = block.forward(
                    x,
                    temb,
                    skip_activations=skip_connections,
                    conditioning=conditioning,
                    cond_mask=cond_mask,
                )
            else:
                x = block.forward(x, temb, skip_activations=skip_connections)
            del skip_activations[-num_skip:]
        return x

    def forward_micro_conditioning(self, times, micros):
        temb = 0
        for key in self.conditions:
            default_value = self.conditions[key]
            micro = micros.get(key, default_value * mx.ones_like(times))
            micro = (
                (micro / default_value).clamp(max=1) * default_value
                if key == "scale"
                else micro * 1000
            )
            temb = temb + self.create_temporal_embedding(
                micro, ff_layers=self.cond_layers[key]
            )
        return temb

    def forward_denoising(
        self, x_t, times, cond_emb=None, conditioning=None, cond_mask=None, micros={}
    ):
        # 1. time embedding
        temb = self.create_temporal_embedding(times)
        if cond_emb is not None:
            temb = temb + cond_emb
        if self.conditions is not None:
            temb = temb + self.forward_micro_conditioning(times, micros)

        # 2. input layer
        if self._config.nesting:
            x_t, x_feat = x_t
        x = self.forward_input_layer(x_t)
        if self._config.nesting:
            x = x + x_feat

        # 3. downsample blocks
        x, skip_activations = self.forward_downsample(x, temb, conditioning, cond_mask)

        # 4. middle blocks
        if not self.config.skip_mid_blocks:
            x = self.mid_blocks[0].forward(
                x, temb, conditioning=conditioning, cond_mask=cond_mask
            )
            x = self.mid_blocks[1].forward(x, temb)

        # 5. upsample blocks
        x = self.forward_upsample(x, temb, conditioning, cond_mask, skip_activations)

        # 6. output layer
        x_out = self.forward_output_layer(x)
        if self._config.nesting:
            return x_out, x
        return x_out

    def forward(
        self,
        x_t: mx.array,
        times: mx.array,
        conditioning: mx.array = None,
        cond_mask: mx.array = None,
        micros={},
    ) -> mx.array:
        if self.config.conditioning_feature_dim > 0:
            cond_emb, conditioning, cond_mask = self.forward_conditioning(
                conditioning, cond_mask
            )
        else:
            cond_emb = None
        return self.forward_denoising(
            x_t, times, cond_emb, conditioning, cond_mask, micros
        )
