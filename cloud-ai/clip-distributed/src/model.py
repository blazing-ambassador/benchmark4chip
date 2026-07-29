"""Mini CLIP image/text tower."""

import torch
import torch.nn as nn


class MiniClip(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.image = nn.Sequential(
            nn.Conv2d(3, dim, 16, stride=16),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim),
        )
        self.text = nn.Sequential(nn.Embedding(49408, dim), nn.Linear(dim, dim))

    def forward(self, images: torch.Tensor, tokens: torch.Tensor):
        img = self.image(images)
        txt = self.text(tokens).mean(dim=1)
        return img, txt
