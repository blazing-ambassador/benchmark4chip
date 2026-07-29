"""Embedding + L2 normalize tower."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingTower(nn.Module):
    def __init__(self, vocab: int = 50000, dim: int = 768):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(ids)
        h = self.proj(h.mean(dim=1))
        return F.normalize(h, dim=-1)
