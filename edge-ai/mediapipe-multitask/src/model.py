"""Serial multi-model MediaPipe-style chain."""

import torch
import torch.nn as nn


class _TinyNet(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))

    def forward(self, x):
        return self.net(x).flatten(1)


class MediaPipeTaskChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.face = _TinyNet(3, 16)
        self.hand = _TinyNet(3, 16)
        self.pose = _TinyNet(3, 16)

    def forward(self, x: torch.Tensor):
        return self.face(x), self.hand(x), self.pose(x)
