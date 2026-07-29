"""FP8 QAT proxy with fake quant."""

import torch
import torch.nn as nn


def fake_fp8(x: torch.Tensor) -> torch.Tensor:
    scale = x.abs().max().clamp(min=1e-6) / 448.0
    q = torch.round(x / scale).clamp(-448, 448)
    return q * scale


class Fp8QatBlock(nn.Module):
    def __init__(self, hidden: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * 4)
        self.fc2 = nn.Linear(hidden * 4, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = fake_fp8(self.fc1(x))
        h = torch.relu(h)
        h = fake_fp8(self.fc2(h))
        return h

    def train_step(self, x: torch.Tensor) -> torch.Tensor:
        y = self.forward(x)
        return y.float().pow(2).mean()
