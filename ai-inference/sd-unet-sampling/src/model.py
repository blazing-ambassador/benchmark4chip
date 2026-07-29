"""Minimal SD-style UNet for iterative sampling."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class MiniSDUNet(nn.Module):
    def __init__(self, channels: int = 4, base: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.pool = nn.AvgPool2d(2)
        self.mid = ConvBlock(base * 2, base * 2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec = ConvBlock(base * 3, base)
        self.out = nn.Conv2d(base, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        m = self.mid(e2)
        d = self.dec(torch.cat([self.up(m), e1], dim=1))
        return self.out(d)

    def estimate_conv_macs(self, batch_size: int, h: int = 32, w: int = 32) -> float:
        # Rough MACs proxy for utilization.
        return batch_size * h * w * 1e6

    def intermediate_bytes(self, batch_size: int, h: int = 32, w: int = 32, dtype_bytes: int = 4) -> int:
        return batch_size * 32 * h * w * dtype_bytes * 3
