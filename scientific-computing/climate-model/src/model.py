"""Climate stencil on large 2D grid."""

import torch
import torch.nn as nn


class ClimateStencilStep(nn.Module):
    def __init__(self, h: int = 256, w: int = 256):
        super().__init__()
        self.grid = nn.Parameter(torch.randn(1, 1, h, w))

    def forward(self) -> torch.Tensor:
        g = self.grid
        lap = (
            g[:, :, 1:-1, 2:]
            + g[:, :, 1:-1, :-2]
            + g[:, :, 2:, 1:-1]
            + g[:, :, :-2, 1:-1]
            - 4 * g[:, :, 1:-1, 1:-1]
        )
        return lap.pow(2).mean()
