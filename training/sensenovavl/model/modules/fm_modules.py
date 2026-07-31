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
    """Decode an H/32 token grid while preserving pretrained RGB weights."""

    def __init__(self, input_dim=4096, hidden_dim=1024, output_channels=3):
        super().__init__()
        if output_channels not in (3, 4):
            raise ValueError(f"ConvDecoder supports 3 or 4 output channels, got {output_channels}.")
        self.output_channels = output_channels
        self.ps1 = nn.PixelShuffle(2)
        self.conv1 = nn.Conv2d(input_dim // 4, hidden_dim, kernel_size=3, padding=1)
        self.act1 = nn.GELU()

        self.ps2 = nn.PixelShuffle(2)
        self.conv2 = nn.Conv2d(hidden_dim // 4, 192, kernel_size=3, padding=1)
        if output_channels == 4:
            self.alpha_conv = nn.Conv2d(hidden_dim // 4, 64, kernel_size=3, padding=1)
            nn.init.zeros_(self.alpha_conv.weight)
            nn.init.zeros_(self.alpha_conv.bias)

        self.ps3 = nn.PixelShuffle(8)

    def forward(self, x):
        x = self.act1(self.conv1(self.ps1(x)))
        x = self.ps2(x)
        rgb = self.ps3(self.conv2(x))
        if self.output_channels == 3:
            return rgb
        alpha = self.ps3(self.alpha_conv(x))
        return torch.cat((rgb, alpha), dim=1)
