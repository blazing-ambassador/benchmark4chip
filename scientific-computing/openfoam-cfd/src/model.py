"""CFD-like dense solve step in FP64."""

import torch
import torch.nn as nn


class CfdDenseStep(nn.Module):
    def __init__(self, n: int = 256):
        super().__init__()
        self.n = n
        self.a = nn.Parameter(torch.randn(n, n) * 0.01)
        self.b = nn.Parameter(torch.randn(n, 1) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.a.double()
        b = self.b.double()
        xd = x.double()
        y = torch.matmul(a, xd) + b
        return y.float()

    def iteration(self, steps: int = 4) -> torch.Tensor:
        x = torch.zeros(self.n, 1)
        for _ in range(steps):
            x = self.forward(x)
        return x
