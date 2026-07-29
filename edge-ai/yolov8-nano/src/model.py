"""YOLOv8-nano style depthwise backbone."""

import torch
import torch.nn as nn


class DWConv(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.pw = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class NanoYoloBackbone(nn.Module):
    def __init__(self, base: int = 16):
        super().__init__()
        self.stem = nn.Conv2d(3, base, 3, stride=2, padding=1)
        self.blocks = nn.Sequential(*[DWConv(base) for _ in range(6)])
        self.head = nn.Conv2d(base, base, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)

    def nms_proxy(self, scores: torch.Tensor, topk: int = 100) -> torch.Tensor:
        vals, idx = torch.topk(scores.reshape(-1), topk)
        return idx
