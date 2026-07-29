"""MD force gather with irregular indexing."""

import torch
import torch.nn as nn


class LammpsForceKernel(nn.Module):
    def __init__(self, atoms: int = 4096, neighbors: int = 32):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(atoms, 3))
        self.register_buffer("idx", torch.randint(0, atoms, (atoms, neighbors)))

    def forward(self) -> torch.Tensor:
        pos = self.pos
        nbr = pos[self.idx]
        diff = nbr - pos.unsqueeze(1)
        dist = diff.pow(2).sum(dim=-1)
        return dist.sum()
