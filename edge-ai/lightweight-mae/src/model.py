"""Lightweight MAE encoder for edge vision."""

import torch
import torch.nn as nn


class LightMaeEncoder(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, 16, stride=16)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True), num_layers=2
        )

    def forward(self, x: torch.Tensor, throttle: float = 1.0) -> torch.Tensor:
        tokens = self.patch(x).flatten(2).transpose(1, 2)
        if throttle < 1.0:
            keep = max(1, int(tokens.size(1) * throttle))
            tokens = tokens[:, :keep, :]
        return self.enc(tokens)
