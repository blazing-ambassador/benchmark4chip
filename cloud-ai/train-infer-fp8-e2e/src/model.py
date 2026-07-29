"""Train then infer FP8 proxy block."""

import torch
import torch.nn as nn


def fake_fp8(x):
    scale = x.abs().max().clamp(min=1e-6) / 448.0
    return torch.round(x / scale).clamp(-448, 448) * scale


class Fp8TrainInferBlock(nn.Module):
    def __init__(self, hidden: int = 512):
        super().__init__()
        self.fc = nn.Linear(hidden, hidden)

    def train_step(self, x: torch.Tensor) -> torch.Tensor:
        y = fake_fp8(self.fc(x))
        return y.float().pow(2).mean()

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return fake_fp8(self.fc(x))
