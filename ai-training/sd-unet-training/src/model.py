"""UNet-style training with backward."""

import torch
import torch.nn as nn


class ConvBN(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.SiLU())

    def forward(self, x):
        return self.net(x)


class MiniUNetTrain(nn.Module):
    def __init__(self, channels: int = 4, base: int = 32):
        super().__init__()
        self.enc = ConvBN(channels, base)
        self.down = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.mid = ConvBN(base * 2, base * 2)
        self.up = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec = ConvBN(base * 2, base)
        self.out = nn.Conv2d(base, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc(x)
        e2 = self.mid(self.down(e1))
        d = self.dec(torch.cat([self.up(e2), e1], dim=1))
        return self.out(d)

    def train_step(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.forward(x)
        return pred.float().pow(2).mean()

    def tensor_count_proxy(self, batch: int, h: int, w: int) -> int:
        return batch * h * w * 12
