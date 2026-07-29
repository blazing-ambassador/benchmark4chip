"""Tensor-parallel sharded linear + simulated collectives."""

import torch
import torch.nn as nn


class ShardedLinearTP(nn.Module):
    def __init__(self, hidden: int = 1024, shards: int = 4):
        super().__init__()
        self.shards = shards
        self.parts = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(shards)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        partials = [p(x) for p in self.parts]
        stacked = torch.stack(partials, dim=0)
        reduced = stacked.sum(dim=0)
        gathered = torch.cat(partials, dim=-1)
        return reduced + gathered[:, :, : x.size(-1)]

    def simulate_allreduce_bytes(self, x: torch.Tensor) -> int:
        return x.numel() * 4 * self.shards
