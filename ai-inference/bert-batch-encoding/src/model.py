"""Minimal BERT encoder for batch embedding inference."""

from typing import Optional

import torch
import torch.nn as nn


class MiniBertEncoder(nn.Module):
    """Tiny BERT-style encoder: token embed + transformer blocks + mean pool."""

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 128,
        intermediate_size: int = 512,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)

        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)

        # TransformerEncoder expects True for masked (ignored) positions.
        key_padding_mask = attention_mask == 0
        hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        hidden = self.layer_norm(hidden)

        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled

    def estimate_bytes_per_forward(self, batch_size: int, seq_len: int, dtype_bytes: int = 4) -> int:
        """Rough activation + weight traffic for on-chip throughput proxy."""
        hidden = self.hidden_size
        layers = len(self.encoder.layers)

        # Weight read (dominant for short-seq inference on chip).
        param_bytes = sum(p.numel() for p in self.parameters()) * dtype_bytes

        # Activation traffic: embed + per-layer hidden states.
        activation_bytes = batch_size * seq_len * hidden * dtype_bytes * (2 + layers * 2)
        return param_bytes + activation_bytes

    def estimate_flops_per_forward(self, batch_size: int, seq_len: int) -> float:
        """Analytic FLOPs estimate for utilization baseline."""
        hidden = self.hidden_size
        layers = len(self.encoder.layers)
        ff = self.encoder.layers[0].linear1.out_features

        # Self-attention: QKV proj + attention score + output proj.
        attn_flops = batch_size * (4 * seq_len * hidden * hidden + 2 * seq_len * seq_len * hidden)
        # FFN: up + down.
        ff_flops = batch_size * seq_len * hidden * ff * 2
        per_layer = attn_flops + ff_flops
        embed_flops = batch_size * seq_len * hidden

        return embed_flops + layers * per_layer
