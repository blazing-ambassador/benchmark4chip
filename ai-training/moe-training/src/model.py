"""MoE training layer with top-k routing."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniMoETrain(nn.Module):
    def __init__(self, hidden: int = 256, experts: int = 8, top_k: int = 2):
        super().__init__()
        self.hidden = hidden
        self.experts = experts
        self.top_k = top_k
        self.router = nn.Linear(hidden, experts)
        self.expert_mlps = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden, hidden * 4), nn.GELU(), nn.Linear(hidden * 4, hidden)) for _ in range(experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, h = x.shape
        flat = x.reshape(b * s, h)
        logits = self.router(flat)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        out = torch.zeros_like(flat)
        for k in range(self.top_k):
            idx = indices[:, k]
            w = weights[:, k].unsqueeze(-1)
            for e in range(self.experts):
                mask = idx == e
                if mask.any():
                    out[mask] = out[mask] + w[mask] * self.expert_mlps[e](flat[mask])
        return out.reshape(b, s, h)

    def train_step(self, x: torch.Tensor) -> torch.Tensor:
        y = self.forward(x)
        return y.float().pow(2).mean()

    def routing_stats(self, x: torch.Tensor):
        flat = x.reshape(-1, self.hidden)
        logits = self.router(flat)
        _, idx = torch.topk(logits, self.top_k, dim=-1)
        counts = torch.bincount(idx.reshape(-1), minlength=self.experts).float()
        balance = (counts.std() / (counts.mean() + 1e-6)).item()
        return balance
