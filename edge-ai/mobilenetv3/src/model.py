"""MobileNetV3-style inverted residuals."""

import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride=stride, padding=1, groups=inp),
            nn.Conv2d(inp, oup, 1),
            nn.Hardswish(),
        )

    def forward(self, x):
        return self.conv(x)


class MobileNetV3Tiny(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            InvertedResidual(16, 24, 2),
            InvertedResidual(24, 24, 1),
            InvertedResidual(24, 40, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(40, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
