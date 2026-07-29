"""Minimal LLaVA-style vision + LLM pipeline."""

import torch
import torch.nn as nn


class MiniVisionTower(nn.Module):
    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size * 4),
        )
        self.hidden_size = hidden_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # [B, num_patches, hidden]
        feats = self.net(images)
        return feats.view(feats.size(0), 4, self.hidden_size)


class MiniLanguageHead(nn.Module):
    def __init__(self, hidden_size: int = 256, vocab_size: int = 32000) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4, dim_feedforward=hidden_size * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(tokens)
        return self.lm_head(hidden[:, -1, :])


class MiniLlavaPipeline(nn.Module):
    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        self.vision = MiniVisionTower(hidden_size)
        self.projector = nn.Linear(hidden_size, hidden_size)
        self.language = MiniLanguageHead(hidden_size)

    def forward_vision(self, images: torch.Tensor) -> torch.Tensor:
        return self.vision(images)

    def forward_project(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        return self.projector(vision_tokens)

    def forward_language(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        return self.language(fused_tokens)

    def forward_e2e(self, images: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        v = self.forward_vision(images)
        p = self.forward_project(v)
        fused = torch.cat([p, text_tokens], dim=1)
        return self.forward_language(fused)

    def feature_bytes(self, batch_size: int, dtype_bytes: int = 4) -> int:
        return batch_size * 4 * self.vision.hidden_size * dtype_bytes
