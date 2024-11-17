# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
""" U-NET architecture."""
import copy
import logging
import math
import pdb
from dataclasses import dataclass, field
from enum import Enum

import einops
from torchinfo import summary

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_mdm import config
from ml_mdm.utils import fix_old_checkpoints


def _fan_in(w):
    return np.prod(w.shape[1:])


def _fan_out(w):
    return w.shape[0]


def _fan_avg(w):
    return 0.5 * (_fan_in(w) + _fan_out(w))


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


@dataclass
class ResNetConfig:
    num_channels: int = field(
        default=-1, metadata={"help": "Number of output channels"}
    )
    output_channels: int = field(
        default=-1, metadata={"help": "Number of output channels"}
    )
    num_groups_norm: int = field(
        default=int(32), metadata={"help": "Number of groups in group norm"}
    )
    dropout: float = field(default=0.0, metadata={"help": "Dropout."})
    use_attention_ffn: bool = field(
        default=False,
        metadata={"help": "Use additional FFN layer after attention layer."},
    )


@config.register_model_config("unet", "unet")
@dataclass
class UNetConfig:
    num_resnets_per_resolution: str = field(
        default="2", metadata={"help": "Number of residual blocks per resolution"}
    )
    temporal_dim: int = field(
        default=None, metadata={"help": "If not set, we use 4 x channels[0]"}
    )
    attention_levels: str = field(
        default="2,3", metadata={"help": "which block to use attention"}
    )
    num_attention_layers: str = field(
        default="1", metadata={"help": "num of attention used for each block"}
    )
    num_temporal_attention_layers: str = field(
        default=None, metadata={"help": "only useful in the temporal mode"}
    )
    conditioning_feature_dim: int = field(
        default=-1,
        metadata={"help": "Dimensions of conditioning vector for cross-attention"},
    )
    conditioning_feature_proj_dim: int = field(
        default=-1,
        metadata={"help": "If value > 0, lieanrly project the conditioning dimension"},
    )
    num_lm_head_layers: int = field(
        default=0,
        metadata={
            "help": "additional learned self attention layers on top of frozen lm embeddings"
        },
    )
    masked_cross_attention: int = field(
        default=1,
        metadata={"help": "whether to use masks when computing cross attention"},
    )
    resolution_channels: str = field(default="128,256,256,512,1024")
    skip_mid_blocks: bool = field(
        default=False, metadata={"help": "do not use mid-blocks"}
    )
    skip_cond_emb: bool = field(
        default=False, metadata={"help": "do not use conditioning projection"}
    )
    nesting: bool = field(
        default=False, metadata={"help": "if I am nesting inside another unet"}
    )
    micro_conditioning: str = field(
        default=None,
        metadata={
            "help": "LABEL:DEFAULT_VALUE. For example, scale:1,watermark_score:0"
        },
    )
    temporal_mode: bool = field(default=False)
    temporal_spatial_ds: bool = field(default=False)
    temporal_positional_encoding: bool = field(default=False)
    resnet_config: ResNetConfig = field(
        default_factory=ResNetConfig, metadata={"help": "Resnet configs"}
    )

    def __post_init__(self):
        if isinstance(self.resolution_channels, str):
            self.resolution_channels = [
                int(x) for x in self.resolution_channels.split(",")
            ]
        if isinstance(self.attention_levels, str):
            if self.attention_levels is None or len(self.attention_levels) == 0:
                self.attention_levels = []
            else:
                self.attention_levels = [
                    int(x) for x in self.attention_levels.split(",")
                ]
        if isinstance(self.num_attention_layers, str):
            self.num_attention_layers = [
                int(x) for x in self.num_attention_layers.split(",")
            ]
            if len(self.num_attention_layers) == 1:
                self.num_attention_layers = self.num_attention_layers * len(
                    self.resolution_channels
                )
            assert len(self.num_attention_layers) == len(self.resolution_channels)
        if isinstance(self.num_resnets_per_resolution, str):
            self.num_resnets_per_resolution = [
                int(x) for x in self.num_resnets_per_resolution.split(",")
            ]
            if len(self.num_resnets_per_resolution) == 1:
                self.num_resnets_per_resolution = self.num_resnets_per_resolution * len(
                    self.resolution_channels
                )
            assert len(self.num_resnets_per_resolution) == len(self.resolution_channels)
        if self.num_temporal_attention_layers and isinstance(
            self.num_temporal_attention_layers, str
        ):
            self.num_temporal_attention_layers = [
                int(x) for x in self.num_temporal_attention_layers.split(",")
            ]


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


class ResNet(nn.Module):
    def __init__(self, time_emb_channels, config: ResNetConfig):
        # TODO(ndjaitly): What about scales of weights.
        super(ResNet, self).__init__()
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
        self.conv2 = zero_module(
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
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        ta, tb = (
            self.time_layer(F.silu(temb)).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        )
        if h.size(0) > ta.size(0):  # HACK. repeat to match the shape.
            N = h.size(0) // ta.size(0)
            ta = einops.repeat(ta, "b c h w -> (b n) c h w", n=N)
            tb = einops.repeat(tb, "b c h w -> (b n) c h w", n=N)
        h = F.silu(self.norm2(h) * (1 + ta) + tb)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.config.output_channels != self.config.num_channels:
            x = self.conv3(x)
        return h + x


class SelfAttention(nn.Module):
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
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.cond_dim = cond_dim
        if cond_dim is not None and cond_dim > 0:
            self.norm_cond = nn.LayerNorm(cond_dim)
            self.kv_cond = nn.Linear(cond_dim, channels * 2)
        self.proj_out = zero_module(nn.Conv2d(channels, channels, 1))
        if use_attention_ffn:
            self.ffn = nn.Sequential(
                nn.GroupNorm(32, channels),
                nn.Conv2d(channels, 4 * channels, 1),
                nn.GELU(),
                zero_module(nn.Conv2d(4 * channels, channels, 1)),
            )
        else:
            self.ffn = None

    def attention(self, q, k, v, mask=None):
        bs, width, length = q.shape
        ch = width // self.num_heads
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
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
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.num_heads, ch, -1))
        return a.reshape(bs, -1, length)

    def forward(self, x, cond=None, cond_mask=None):
        # assert (self.cond_dim is not None) == (cond is not None)
        b, c, *spatial = x.shape
        # x = x.reshape(b, c, -1)
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


class SelfAttention1D(nn.Module):
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
        self.proj_out = zero_module(nn.Linear(channels, channels))
        if use_attention_ffn:
            self.ffn = nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, 4 * channels),
                nn.GELU(),
                zero_module(nn.Linear(4 * channels, channels)),
            )
        else:
            self.ffn = None
        if pos_emb:
            from rotary_embedding_torch import RotaryEmbedding

            self.pos_emb = RotaryEmbedding(dim=channels // self.num_heads)
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
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        if mask is not None:
            mask = mask.view(mask.size(0), 1, 1, mask.size(1))
            weight = weight.masked_fill(mask == 0, float("-inf"))
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum(
            "bhts,bshc->bthc", weight, v.reshape(bs, -1, self.num_heads, ch)
        )
        return a.reshape(bs, length, -1)

    def forward(self, x, mask):
        # assert (self.cond_dim is not None) == (cond is not None)
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=-1)
        h = self.attention(q, k, v, mask)
        h = self.proj_out(h)
        x = x + h
        if self.ffn is not None:
            x = x + self.ffn(x)
        return x


class TemporalAttentionBlock(nn.Module):
    def __init__(
        self, channels, num_heads=8, num_head_channels=-1, down=False, pos_emb=False
    ):
        super().__init__()
        self.attn = SelfAttention1D(
            channels, num_heads, num_head_channels, pos_emb=pos_emb
        )
        self.mlp = MLP(channels, multiplier=4)
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
            x = self.down_conv(x)

        T, H, W = x.size(0) // temb.size(0), x.size(2), x.size(3)
        x = einops.rearrange(x, "(b t) c h w -> (b h w) t c", t=T)
        x = self.mlp(self.attn(x, None))
        x = einops.rearrange(x, "(b h w) t c -> (b t) c h w", h=H, w=W)

        if self.down:
            x = self.up_conv(F.interpolate(x, scale_factor=2))
        x = x + x_
        return x


class MLP(nn.Module):
    def __init__(self, channels, multiplier=4):
        super().__init__()
        self.main = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, multiplier * channels),
            nn.GELU(),
            zero_module(nn.Linear(multiplier * channels, channels)),
        )

    def forward(self, x):
        return x + self.main(x)


class SelfAttention1DBlock(nn.Module):
    def __init__(self, channels, num_heads=8, num_head_channels=-1, mlp_multiplier=4):
        super().__init__()
        self.attn = SelfAttention1D(channels, num_heads, num_head_channels)
        self.mlp = MLP(channels, mlp_multiplier)

    def forward(self, x, mask):
        return self.mlp(self.attn(x, mask))


class ResNetBlock(nn.Module):
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
            resnets.append(ResNet(temporal_dim, cur_config))
        self.resnets = nn.ModuleList(resnets)

        if self.num_attention_layers > 0:
            attn = []
            for i in range(num_residual_blocks):
                for j in range(self.num_attention_layers):
                    attn.append(
                        SelfAttention(
                            resnet_configs[i].output_channels,
                            cond_dim=conditioning_feature_dim,
                            use_attention_ffn=resnet_configs[i].use_attention_ffn,
                        )
                    )
            self.attn = nn.ModuleList(attn)

        if (
            self.num_temporal_attention_layers
            and self.num_temporal_attention_layers > 0
            and (not self.temporal_spatial_ds)
        ):
            t_attn = []
            for i in range(num_residual_blocks):
                for j in range(self.num_temporal_attention_layers):
                    t_attn.append(
                        TemporalAttentionBlock(
                            resnet_configs[i].output_channels,
                            num_head_channels=32,
                            down=True,
                            pos_emb=temporal_pos_emb,
                        )
                    )
            self.t_attn = nn.ModuleList(t_attn)

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
                x = torch.cat((x, skip_input), axis=1)

            x = self.resnets[i](x, temb)
            if self.num_attention_layers > 0:
                L = self.num_attention_layers
                for j in range(L):
                    x = self.attn[i * L + j](x, conditioning, cond_mask)
            if (
                self.num_temporal_attention_layers
                and self.num_temporal_attention_layers > 0
            ):
                L = self.num_temporal_attention_layers
                for j in range(L):
                    x = self.t_attn[i * L + j](x, temb)
            activations.append(x)

        if self.downsample_output or self.upsample_output:
            if self.temporal and (not self.temporal_spatial_ds):
                T, H, W = x.size(0) // temb.size(0), x.size(2), x.size(3)
                x = einops.rearrange(x, "(b t) c h w -> (b h w) c t", t=T)
            if self.upsample_output:
                x = F.interpolate(x.type(torch.float32), scale_factor=2).type(x.dtype)
            x = self.resample(x)
            if self.temporal and (not self.temporal_spatial_ds):
                x = einops.rearrange(x, "(b h w) c t -> (b t) c h w", h=H, w=W)
            activations.append(x)

        if not return_activations:
            return x
        return x, activations


@config.register_model("unet")
class UNet(nn.Module):
    def __init__(self, input_channels, output_channels, config: UNetConfig):
        super().__init__()
        self.down_blocks = []
        self.config = config
        self.input_channels = input_channels
        self.output_channels = output_channels
        # we will overwrite config.conditioning_feature_dim if config.conditioning_feature_proj_dim is provided
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
        emb = math.log(10000) / half_dim  # make this consistent to the adm unet
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        self.register_buffer("t_emb", emb.unsqueeze(0), persistent=False)

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
            cond_layers = {}
            for condition in self.conditions:
                cond_layers[condition] = nn.ModuleList(
                    [
                        nn.Linear(self.temporal_dim // 4, self.temporal_dim),
                        zero_module(nn.Linear(self.temporal_dim, self.temporal_dim)),
                    ]
                )
            self.cond_layers = nn.ModuleDict(cond_layers)

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
                ResNetBlock(
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
                ResNetBlock(
                    self.temporal_dim,
                    1,
                    True,  # attn
                    False,  # downsample
                    False,  # upsample
                    resnet_configs=[resnet_config],
                    conditioning_feature_dim=config.conditioning_feature_dim,
                ),
                ResNetBlock(
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
                ResNetBlock(
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
        self.conv_out = zero_module(
            nn.Conv2d(channels, output_channels, kernel_size=3, padding=1)
        )
        self._config = config
        self.down_blocks = nn.ModuleList(self.down_blocks)
        if not config.skip_mid_blocks:
            self.mid_blocks = nn.ModuleList(self.mid_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)

        self.masked_cross_attention = config.masked_cross_attention
        if config.conditioning_feature_dim > 0 and (not config.skip_cond_emb):
            if config.conditioning_feature_proj_dim > 0:
                # note that now config.conditioning_feature_proj_dim == config.conditioning_feature_dim
                self.lm_proj = nn.Linear(
                    self.input_conditioning_feature_dim, config.conditioning_feature_dim
                )
            self.lm_head = nn.ModuleList(
                [
                    SelfAttention1DBlock(config.conditioning_feature_dim)
                    for _ in range(config.num_lm_head_layers)
                ]
            )

        self.is_temporal = []

    @property
    def model_type(self):
        return "unet"

    def print_size(self, target_image_size=64):
        summary(
            self,
            [
                (1, self.input_channels, target_image_size, target_image_size),  # x_t
                (1,),  # times
                (1, 32, self.input_conditioning_feature_dim),  # conditioning
                (1, 32),
            ],  # condition_mask
            dtypes=[torch.float, torch.float, torch.float, torch.float],
            col_names=["input_size", "output_size", "num_params"],
            row_settings=["var_names"],
            depth=4,
        )

    def save(self, fname, other_items=None):
        logging.info(f"Saving model file: {fname}")
        checkpoint = {"state_dict": self.state_dict()}
        if other_items is not None:
            for k, v in other_items.items():
                checkpoint[k] = v
        torch.save(checkpoint, fname)

    def load(self, fname):
        logging.info(f"Loading model file: {fname}")
        fix_old_checkpoints.mimic_old_modules()
        # first load to cpu or we will run out of memory.
        checkpoint = torch.load(fname, map_location=lambda storage, loc: storage)
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
        temb = times.view(-1, 1) * self.t_emb
        temb = torch.cat([torch.sin(temb), torch.cos(temb)], dim=1)
        if temb.shape[1] % 2 == 1:
            # zero pad
            temb = torch.cat([temb, torch.zeros(times.shape[0], 1)], dim=1)
        if ff_layers is None:
            layer1, layer2 = self.temb_layer1, self.temb_layer2
        else:
            layer1, layer2 = ff_layers
        temb = layer2(F.silu(layer1(temb)))
        return temb

    def forward_conditioning(self, conditioning, cond_mask):
        if self.config.conditioning_feature_proj_dim > 0:
            conditioning = self.lm_proj(conditioning)
        for head in self.lm_head:
            conditioning = head(
                conditioning, mask=cond_mask if self.masked_cross_attention else None
            )
        if cond_mask is None or (
            not self.masked_cross_attention and len(self.lm_head) > 0
        ):
            y = conditioning.mean(dim=1)
        else:
            y = (cond_mask.unsqueeze(-1) * conditioning).sum(dim=1) / cond_mask.sum(
                dim=1, keepdim=True
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
        x_out = F.silu(self.norm_out(x))
        x_out = self.conv_out(x_out)
        return x_out

    @temporal_wrapper
    def forward_downsample(self, x, temb, conditioning, cond_mask):
        skip_activations = [x]
        for i, block in enumerate(self.down_blocks):
            if i in self.config.attention_levels:
                x, activations = block(
                    x,
                    temb,
                    return_activations=True,
                    conditioning=conditioning,
                    cond_mask=cond_mask,
                )
            else:
                x, activations = block(x, temb, return_activations=True)
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
                x = block(
                    x,
                    temb,
                    skip_activations=skip_connections,
                    conditioning=conditioning,
                    cond_mask=cond_mask,
                )
            else:
                x = block(x, temb, skip_activations=skip_connections)
            del skip_activations[-num_skip:]
        return x

    def forward_micro_conditioning(self, times, micros):
        temb = 0
        for key in self.conditions:
            default_value = self.conditions[key]
            micro = micros.get(key, default_value * torch.ones_like(times))
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
            x = self.mid_blocks[0](
                x, temb, conditioning=conditioning, cond_mask=cond_mask
            )
            x = self.mid_blocks[1](x, temb)

        # 5. upsample blocks
        x = self.forward_upsample(x, temb, conditioning, cond_mask, skip_activations)

        # 6. output layer
        x_out = self.forward_output_layer(x)
        if self._config.nesting:
            return x_out, x
        return x_out

    def forward(
        self,
        x_t: torch.Tensor,
        times: torch.Tensor,
        conditioning: torch.Tensor = None,
        cond_mask: torch.Tensor = None,
        micros={},
    ) -> torch.Tensor:
        if self.config.conditioning_feature_dim > 0:
            cond_emb, conditioning, cond_mask = self.forward_conditioning(
                conditioning, cond_mask
            )
        else:
            cond_emb = None
        return self.forward_denoising(
            x_t, times, cond_emb, conditioning, cond_mask, micros
        )
