"""Minimal dense LM training block (forward + backward proxy)."""

from typing import Tuple

import torch
import torch.nn as nn


class MiniLmTrainBlock(nn.Module):
    def __init__(self, vocab: int = 32000, hidden: int = 512, layers: int = 4, heads: int = 8, ff: int = 2048):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        enc = nn.TransformerEncoderLayer(hidden, heads, ff, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embed(tokens)
        h = self.encoder(h)
        return self.lm_head(h)

    def train_step(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.forward(tokens)
        loss = logits.float().pow(2).mean()
        return loss

    def estimate_bytes(self, batch: int, seq: int, dtype_bytes: int = 2) -> int:
        params = sum(p.numel() for p in self.parameters()) * dtype_bytes
        act = batch * seq * 512 * dtype_bytes * 8
        return params + act

    def estimate_flops(self, batch: int, seq: int) -> float:
        return batch * seq * 512 * 512 * 12.0
