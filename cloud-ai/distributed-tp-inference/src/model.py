"""Sharded LM inference with collective proxy."""

import torch
import torch.nn as nn


class ShardedLmInference(nn.Module):
    def __init__(self, vocab: int = 32000, hidden: int = 512, shards: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.shards = shards
        self.projections = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(shards)])
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embed(tokens)
        parts = [p(h) for p in self.projections]
        merged = torch.stack(parts, dim=0).mean(dim=0)
        return self.head(merged)

    def collective_bytes(self, tokens: torch.Tensor) -> int:
        return tokens.numel() * 4 * self.shards
