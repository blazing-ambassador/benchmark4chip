"""Tiny DistilBERT for edge memory stress."""

import torch
import torch.nn as nn


class DistilBertTiny(nn.Module):
    def __init__(self, vocab: int = 30522, hidden: int = 128, layers: int = 2, heads: int = 2):
        super().__init__()
        self.hidden = hidden
        self.embed = nn.Embedding(vocab, hidden)
        enc = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(ids)
        h = self.encoder(h)
        return self.out(h)

    def chunked_attention_proxy(self, ids: torch.Tensor, chunk: int = 16) -> torch.Tensor:
        outs = []
        for i in range(0, ids.size(1), chunk):
            sl = ids[:, i : i + chunk]
            outs.append(self.forward(sl))
        return torch.cat(outs, dim=1)
