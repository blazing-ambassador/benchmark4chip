"""NPB-style sustained FP kernel."""

import torch
import torch.nn as nn


class NpbComputeKernel(nn.Module):
    def __init__(self, n: int = 512):
        super().__init__()
        self.n = n
        self.u = nn.Parameter(torch.randn(n, n))

    def forward(self) -> torch.Tensor:
        u = self.u
        v = torch.fft.fft2(u)
        w = torch.real(v * torch.conj(v))
        return w.sum()
