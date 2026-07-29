"""Minimal INT8/FP8 quantized inference stack."""

import torch
import torch.nn as nn


def quantize_int8(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    q = torch.round(x / scale + zero_point)
    return torch.clamp(q, -128, 127)


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    return (q - zero_point) * scale


def quantize_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    # Simulated FP8 dynamic range clamp for benchmark proxy.
    return torch.clamp(x, -448.0, 448.0)


class MiniQuantMLP(nn.Module):
    def __init__(self, dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.act = nn.GELU()

    def forward_fp32(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

    def forward_int8(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.abs().max() / 127.0 + 1e-8
        zp = torch.zeros((), device=x.device)
        qx = quantize_int8(x, scale, zp)
        x_dq = dequantize_int8(qx, scale, zp)
        w1 = self.fc1.weight
        sw = w1.abs().max() / 127.0 + 1e-8
        qw = quantize_int8(w1, sw, zp)
        w_dq = dequantize_int8(qw, sw, zp)
        h = torch.matmul(x_dq, w_dq.t()) + self.fc1.bias
        h = self.act(h)
        return self.fc2(h)

    def forward_fp8(self, x: torch.Tensor) -> torch.Tensor:
        x8 = quantize_fp8_e4m3(x)
        w8 = quantize_fp8_e4m3(self.fc1.weight)
        h = torch.matmul(x8, w8.t()) + self.fc1.bias
        h = self.act(quantize_fp8_e4m3(h))
        return self.fc2(h)

    def param_bytes(self, dtype_bytes: int = 4) -> int:
        return sum(p.numel() for p in self.parameters()) * dtype_bytes
