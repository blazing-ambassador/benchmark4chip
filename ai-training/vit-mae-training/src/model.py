"""ViT-MAE style masked patch training."""

import torch
import torch.nn as nn


class MiniViTMae(nn.Module):
    def __init__(self, patch: int = 8, dim: int = 256, depth: int = 4, heads: int = 4):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(3, dim, patch, stride=patch)
        enc = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=depth)
        self.decoder = nn.Linear(dim, patch * patch * 3)

    def forward(self, images: torch.Tensor, mask_ratio: float = 0.75) -> torch.Tensor:
        b, c, h, w = images.shape
        tokens = self.proj(images).flatten(2).transpose(1, 2)
        n = tokens.shape[1]
        n_mask = int(n * mask_ratio)
        perm = torch.rand(b, n, device=images.device).argsort(dim=1)
        masked_idx = perm[:, :n_mask]
        visible_idx = perm[:, n_mask:]
        visible = tokens.gather(1, visible_idx.unsqueeze(-1).expand(-1, -1, tokens.size(-1)))
        encoded = self.encoder(visible)
        recon = self.decoder(encoded)
        return recon

    def train_step(self, images: torch.Tensor) -> torch.Tensor:
        out = self.forward(images)
        return out.float().pow(2).mean()
