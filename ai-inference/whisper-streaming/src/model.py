"""Minimal Whisper-style streaming encoder."""

import torch
import torch.nn as nn


class MiniWhisperEncoder(nn.Module):
    def __init__(self, n_mels: int = 80, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.n_mels = n_mels
        self.hidden = hidden
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [B, n_mels, T]
        x = self.conv(mel).transpose(1, 2)
        return self.encoder(x)

    def context_bytes(self, batch_size: int, total_frames: int, dtype_bytes: int = 4) -> int:
        return batch_size * total_frames * self.hidden * dtype_bytes
