"""INT4 weight proxy via packed dequant matmul."""

import torch
import torch.nn as nn


class Int4LinearStack(nn.Module):
    def __init__(self, hidden: int = 512, layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.scales = nn.ParameterList([nn.Parameter(torch.ones(hidden)) for _ in range(layers)])

    def dequant_weights(self, layer_idx: int) -> torch.Tensor:
        w = self.layers[layer_idx].weight
        scale = self.scales[layer_idx].unsqueeze(1)
        q = torch.round(w / scale).clamp(-8, 7)
        return q * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, lin in enumerate(self.layers):
            w = self.dequant_weights(i)
            x = torch.nn.functional.linear(x, w, lin.bias)
            x = torch.relu(x)
        return x
