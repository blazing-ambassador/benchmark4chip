"""MoE batch inference."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEInferenceBatch(nn.Module):
    def __init__(self, hidden: int = 256, experts: int = 16, top_k: int = 2):
        super().__init__()
        self.router = nn.Linear(hidden, experts)
        self.experts = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, h = x.shape
        flat = x.reshape(b * s, h)
        logits = self.router(flat)
        w, idx = torch.topk(logits, 2, dim=-1)
        w = F.softmax(w, dim=-1)
        out = torch.zeros_like(flat)
        for k in range(2):
            for e, expert in enumerate(self.experts):
                mask = idx[:, k] == e
                if mask.any():
                    out[mask] += w[mask, k].unsqueeze(1) * expert(flat[mask])
        return out.reshape(b, s, h)
