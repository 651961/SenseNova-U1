# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Sinusoidal timestep embedding follows standard diffusion / flow-matching
# practice, e.g. DiT (Peebles & Xie, 2022) and FLUX (Black Forest Labs).
# Reference: https://github.com/facebookresearch/DiT
import math

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class ConvDecoder(nn.Module):
    """Decode an H/32 token grid with one RGB/RGBA output convolution."""

    final_upscale = 8
    rgb_channels = 3

    def __init__(self, input_dim=4096, hidden_dim=1024, output_channels=3):
        super().__init__()
        if output_channels not in (3, 4):
            raise ValueError(f"ConvDecoder supports 3 or 4 output channels, got {output_channels}.")
        self.output_channels = output_channels
        self.ps1 = nn.PixelShuffle(2)
        self.conv1 = nn.Conv2d(input_dim // 4, hidden_dim, kernel_size=3, padding=1)
        self.act1 = nn.GELU()

        self.ps2 = nn.PixelShuffle(2)
        output_subpixel_channels = output_channels * self.final_upscale**2
        self.conv2 = nn.Conv2d(
            hidden_dim // 4,
            output_subpixel_channels,
            kernel_size=3,
            padding=1,
        )
        if output_channels == 4:
            alpha_start = self.rgb_channels * self.final_upscale**2
            nn.init.zeros_(self.conv2.weight[alpha_start:])
            nn.init.zeros_(self.conv2.bias[alpha_start:])

        self.ps3 = nn.PixelShuffle(self.final_upscale)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Expand legacy RGB/split-RGBA heads into the combined convolution."""

        weight_key = f"{prefix}conv2.weight"
        bias_key = f"{prefix}conv2.bias"
        alpha_weight_key = f"{prefix}alpha_conv.weight"
        alpha_bias_key = f"{prefix}alpha_conv.bias"
        rgb_subpixel_channels = self.rgb_channels * self.final_upscale**2

        source_weight = state_dict.get(weight_key)
        if (
            self.output_channels == 4
            and source_weight is not None
            and source_weight.shape[0] == rgb_subpixel_channels
        ):
            expanded_weight = source_weight.new_zeros(self.conv2.weight.shape)
            expanded_weight[:rgb_subpixel_channels].copy_(source_weight)
            legacy_alpha_weight = state_dict.pop(alpha_weight_key, None)
            if legacy_alpha_weight is not None:
                if legacy_alpha_weight.shape != expanded_weight[rgb_subpixel_channels:].shape:
                    error_msgs.append(
                        f"size mismatch for {alpha_weight_key}: expected "
                        f"{tuple(expanded_weight[rgb_subpixel_channels:].shape)}, got "
                        f"{tuple(legacy_alpha_weight.shape)}"
                    )
                else:
                    expanded_weight[rgb_subpixel_channels:].copy_(legacy_alpha_weight)
            state_dict[weight_key] = expanded_weight

            source_bias = state_dict.get(bias_key)
            if source_bias is not None and source_bias.shape[0] == rgb_subpixel_channels:
                expanded_bias = source_bias.new_zeros(self.conv2.bias.shape)
                expanded_bias[:rgb_subpixel_channels].copy_(source_bias)
                legacy_alpha_bias = state_dict.pop(alpha_bias_key, None)
                if legacy_alpha_bias is not None:
                    if legacy_alpha_bias.shape != expanded_bias[rgb_subpixel_channels:].shape:
                        error_msgs.append(
                            f"size mismatch for {alpha_bias_key}: expected "
                            f"{tuple(expanded_bias[rgb_subpixel_channels:].shape)}, got "
                            f"{tuple(legacy_alpha_bias.shape)}"
                        )
                    else:
                        expanded_bias[rgb_subpixel_channels:].copy_(legacy_alpha_bias)
                state_dict[bias_key] = expanded_bias

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        x = self.act1(self.conv1(self.ps1(x)))
        x = self.ps2(x)
        return self.ps3(self.conv2(x))
