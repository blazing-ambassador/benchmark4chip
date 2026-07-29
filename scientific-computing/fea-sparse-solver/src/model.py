"""Sparse SpMV step for FEA."""

import torch
import torch.nn as nn


class SparseFeAStep(nn.Module):
    def __init__(self, n: int = 5000, nnz: int = 50000):
        super().__init__()
        self.register_buffer("rows", torch.randint(0, n, (nnz,)))
        self.register_buffer("cols", torch.randint(0, n, (nnz,)))
        self.register_buffer("vals", torch.randn(nnz))
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros(self.n, device=x.device, dtype=x.dtype)
        y.index_add_(0, self.rows, self.vals * x[self.cols])
        return y
