"""3D FFT workload."""

import torch
import torch.nn as nn


class Fft3dWorkload(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.fft.fftn(x)
        return torch.real(y)
