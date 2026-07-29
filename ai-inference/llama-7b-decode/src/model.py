"""Minimal LLaMA-style incremental decoder with KV cache."""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def _attend(self, q, k, v, causal: bool) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if causal and q.size(-2) > 1:
            t = q.size(-2)
            mask = torch.triu(torch.ones(t, t, device=q.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    def forward(
        self,
        hidden: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.norm1(hidden)
        q = self.q_proj(x).view(hidden.size(0), hidden.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_proj(x).view(hidden.size(0), hidden.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(x).view(hidden.size(0), hidden.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k_new], dim=2)
            v = torch.cat([past_kv[1], v_new], dim=2)
        else:
            k, v = k_new, v_new

        causal = past_kv is None
        attn = self._attend(q, k, v, causal=causal)
        attn = attn.transpose(1, 2).contiguous().view(hidden.size(0), hidden.size(1), -1)
        hidden = hidden + self.o_proj(attn)
        hidden = hidden + self.mlp(self.norm2(hidden))
        return hidden, (k, v)


class MiniLlamaDecoder(nn.Module):
    """Tiny decoder for KV-cache incremental inference."""

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        max_ctx: int = 512,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_ctx = max_ctx
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(MiniDecoderLayer(hidden_size, num_heads) for _ in range(num_layers))
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_kvs: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        hidden = self.embed(input_ids)
        new_kvs = []
        for i, layer in enumerate(self.layers):
            past = None if past_kvs is None else past_kvs[i]
            hidden, kv = layer(hidden, past)
            new_kvs.append(kv)
        logits = self.lm_head(hidden[:, -1:, :])
        return logits, new_kvs

    def kv_cache_bytes(self, batch_size: int, seq_len: int, dtype_bytes: int = 4) -> int:
        per_layer = 2 * batch_size * seq_len * self.hidden_size * dtype_bytes
        return self.num_layers * per_layer

    def weight_bytes(self, dtype_bytes: int = 4) -> int:
        return sum(p.numel() for p in self.parameters()) * dtype_bytes
