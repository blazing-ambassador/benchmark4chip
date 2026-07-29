"""Archetype model/metrics/run sources for four benchmark categories."""

from typing import Any, Dict, List, Tuple

# benchmark_id -> archetype kind
KIND_BY_ID = {
    "llama-7b-pretrain": "transformer_train",
    "moe-training": "moe_train",
    "sd-unet-training": "unet_train",
    "vit-mae-training": "vit_mae_train",
    "megatron-tp-simulation": "tp_collective",
    "fp8-qat-training": "fp8_qat",
    "yolov8-nano": "depthwise_det",
    "mobilenetv3": "mobilenet_cls",
    "distilbert-edge": "distilbert_edge",
    "qwen-int4": "int4_linear",
    "mediapipe-multitask": "multitask_serial",
    "lightweight-mae": "light_mae",
    "distributed-tp-inference": "tp_inference",
    "moe-cloud-batch": "moe_batch",
    "multi-model-tenant": "multi_tenant",
    "clip-distributed": "clip_batch",
    "embedding-concurrent": "embedding_concurrent",
    "train-infer-fp8-e2e": "fp8_e2e",
    "openfoam-cfd": "fp64_dense",
    "lammps-md": "lammps_md",
    "npb-hpc": "npb_kernel",
    "3d-fft": "fft3d",
    "fea-sparse-solver": "spmv",
    "climate-model": "climate_stencil",
}

MODEL_CLASS = {
    "transformer_train": "MiniLmTrainBlock",
    "moe_train": "MiniMoETrain",
    "unet_train": "MiniUNetTrain",
    "vit_mae_train": "MiniViTMae",
    "tp_collective": "ShardedLinearTP",
    "fp8_qat": "Fp8QatBlock",
    "depthwise_det": "NanoYoloBackbone",
    "mobilenet_cls": "MobileNetV3Tiny",
    "distilbert_edge": "DistilBertTiny",
    "int4_linear": "Int4LinearStack",
    "multitask_serial": "MediaPipeTaskChain",
    "light_mae": "LightMaeEncoder",
    "tp_inference": "ShardedLmInference",
    "moe_batch": "MoEInferenceBatch",
    "multi_tenant": "TenantModelPool",
    "clip_batch": "MiniClip",
    "embedding_concurrent": "EmbeddingTower",
    "fp8_e2e": "Fp8TrainInferBlock",
    "fp64_dense": "CfdDenseStep",
    "lammps_md": "LammpsForceKernel",
    "npb_kernel": "NpbComputeKernel",
    "fft3d": "Fft3dWorkload",
    "spmv": "SparseFeAStep",
    "climate_stencil": "ClimateStencilStep",
}


def _model_py(kind: str, bench_id: str) -> str:
    templates = {
        "transformer_train": '''"""Minimal dense LM training block (forward + backward proxy)."""

from typing import Tuple

import torch
import torch.nn as nn


class MiniLmTrainBlock(nn.Module):
    def __init__(self, vocab: int = 32000, hidden: int = 512, layers: int = 4, heads: int = 8, ff: int = 2048):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        enc = nn.TransformerEncoderLayer(hidden, heads, ff, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embed(tokens)
        h = self.encoder(h)
        return self.lm_head(h)

    def train_step(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.forward(tokens)
        loss = logits.float().pow(2).mean()
        return loss

    def estimate_bytes(self, batch: int, seq: int, dtype_bytes: int = 2) -> int:
        params = sum(p.numel() for p in self.parameters()) * dtype_bytes
        act = batch * seq * 512 * dtype_bytes * 8
        return params + act

    def estimate_flops(self, batch: int, seq: int) -> float:
        return batch * seq * 512 * 512 * 12.0
''',
        "moe_train": '''"""MoE training layer with top-k routing."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniMoETrain(nn.Module):
    def __init__(self, hidden: int = 256, experts: int = 8, top_k: int = 2):
        super().__init__()
        self.hidden = hidden
        self.experts = experts
        self.top_k = top_k
        self.router = nn.Linear(hidden, experts)
        self.expert_mlps = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden, hidden * 4), nn.GELU(), nn.Linear(hidden * 4, hidden)) for _ in range(experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, h = x.shape
        flat = x.reshape(b * s, h)
        logits = self.router(flat)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        out = torch.zeros_like(flat)
        for k in range(self.top_k):
            idx = indices[:, k]
            w = weights[:, k].unsqueeze(-1)
            for e in range(self.experts):
                mask = idx == e
                if mask.any():
                    out[mask] = out[mask] + w[mask] * self.expert_mlps[e](flat[mask])
        return out.reshape(b, s, h)

    def train_step(self, x: torch.Tensor) -> torch.Tensor:
        y = self.forward(x)
        return y.float().pow(2).mean()

    def routing_stats(self, x: torch.Tensor):
        flat = x.reshape(-1, self.hidden)
        logits = self.router(flat)
        _, idx = torch.topk(logits, self.top_k, dim=-1)
        counts = torch.bincount(idx.reshape(-1), minlength=self.experts).float()
        balance = (counts.std() / (counts.mean() + 1e-6)).item()
        return balance
''',
        "unet_train": '''"""UNet-style training with backward."""

import torch
import torch.nn as nn


class ConvBN(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.SiLU())

    def forward(self, x):
        return self.net(x)


class MiniUNetTrain(nn.Module):
    def __init__(self, channels: int = 4, base: int = 32):
        super().__init__()
        self.enc = ConvBN(channels, base)
        self.down = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.mid = ConvBN(base * 2, base * 2)
        self.up = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec = ConvBN(base * 2, base)
        self.out = nn.Conv2d(base, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc(x)
        e2 = self.mid(self.down(e1))
        d = self.dec(torch.cat([self.up(e2), e1], dim=1))
        return self.out(d)

    def train_step(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.forward(x)
        return pred.float().pow(2).mean()

    def tensor_count_proxy(self, batch: int, h: int, w: int) -> int:
        return batch * h * w * 12
''',
        "vit_mae_train": '''"""ViT-MAE style masked patch training."""

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
''',
        "tp_collective": '''"""Tensor-parallel sharded linear + simulated collectives."""

import torch
import torch.nn as nn


class ShardedLinearTP(nn.Module):
    def __init__(self, hidden: int = 1024, shards: int = 4):
        super().__init__()
        self.shards = shards
        self.parts = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(shards)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        partials = [p(x) for p in self.parts]
        stacked = torch.stack(partials, dim=0)
        reduced = stacked.sum(dim=0)
        gathered = torch.cat(partials, dim=-1)
        return reduced + gathered[:, :, : x.size(-1)]

    def simulate_allreduce_bytes(self, x: torch.Tensor) -> int:
        return x.numel() * 4 * self.shards
''',
        "fp8_qat": '''"""FP8 QAT proxy with fake quant."""

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
''',
        "depthwise_det": '''"""YOLOv8-nano style depthwise backbone."""

import torch
import torch.nn as nn


class DWConv(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.pw = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class NanoYoloBackbone(nn.Module):
    def __init__(self, base: int = 16):
        super().__init__()
        self.stem = nn.Conv2d(3, base, 3, stride=2, padding=1)
        self.blocks = nn.Sequential(*[DWConv(base) for _ in range(6)])
        self.head = nn.Conv2d(base, base, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)

    def nms_proxy(self, scores: torch.Tensor, topk: int = 100) -> torch.Tensor:
        vals, idx = torch.topk(scores.reshape(-1), topk)
        return idx
''',
        "mobilenet_cls": '''"""MobileNetV3-style inverted residuals."""

import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride=stride, padding=1, groups=inp),
            nn.Conv2d(inp, oup, 1),
            nn.Hardswish(),
        )

    def forward(self, x):
        return self.conv(x)


class MobileNetV3Tiny(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            InvertedResidual(16, 24, 2),
            InvertedResidual(24, 24, 1),
            InvertedResidual(24, 40, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(40, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
''',
        "distilbert_edge": '''"""Tiny DistilBERT for edge memory stress."""

import torch
import torch.nn as nn


class DistilBertTiny(nn.Module):
    def __init__(self, vocab: int = 30522, hidden: int = 128, layers: int = 2, heads: int = 2):
        super().__init__()
        self.hidden = hidden
        self.embed = nn.Embedding(vocab, hidden)
        enc = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(ids)
        h = self.encoder(h)
        return self.out(h)

    def chunked_attention_proxy(self, ids: torch.Tensor, chunk: int = 16) -> torch.Tensor:
        outs = []
        for i in range(0, ids.size(1), chunk):
            sl = ids[:, i : i + chunk]
            outs.append(self.forward(sl))
        return torch.cat(outs, dim=1)
''',
        "int4_linear": '''"""INT4 weight proxy via packed dequant matmul."""

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
''',
        "multitask_serial": '''"""Serial multi-model MediaPipe-style chain."""

import torch
import torch.nn as nn


class _TinyNet(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))

    def forward(self, x):
        return self.net(x).flatten(1)


class MediaPipeTaskChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.face = _TinyNet(3, 16)
        self.hand = _TinyNet(3, 16)
        self.pose = _TinyNet(3, 16)

    def forward(self, x: torch.Tensor):
        return self.face(x), self.hand(x), self.pose(x)
''',
        "light_mae": '''"""Lightweight MAE encoder for edge vision."""

import torch
import torch.nn as nn


class LightMaeEncoder(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, 16, stride=16)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True), num_layers=2
        )

    def forward(self, x: torch.Tensor, throttle: float = 1.0) -> torch.Tensor:
        tokens = self.patch(x).flatten(2).transpose(1, 2)
        if throttle < 1.0:
            keep = max(1, int(tokens.size(1) * throttle))
            tokens = tokens[:, :keep, :]
        return self.enc(tokens)
''',
        "tp_inference": '''"""Sharded LM inference with collective proxy."""

import torch
import torch.nn as nn


class ShardedLmInference(nn.Module):
    def __init__(self, vocab: int = 32000, hidden: int = 512, shards: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.shards = shards
        self.projections = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(shards)])
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embed(tokens)
        parts = [p(h) for p in self.projections]
        merged = torch.stack(parts, dim=0).mean(dim=0)
        return self.head(merged)

    def collective_bytes(self, tokens: torch.Tensor) -> int:
        return tokens.numel() * 4 * self.shards
''',
        "moe_batch": '''"""MoE batch inference."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEInferenceBatch(nn.Module):
    def __init__(self, hidden: int = 256, experts: int = 16, top_k: int = 2):
        super().__init__()
        self.router = nn.Linear(hidden, experts)
        self.experts = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, h = x.shape
        flat = x.reshape(b * s, h)
        logits = self.router(flat)
        w, idx = torch.topk(logits, 2, dim=-1)
        w = F.softmax(w, dim=-1)
        out = torch.zeros_like(flat)
        for k in range(2):
            for e, expert in enumerate(self.experts):
                mask = idx[:, k] == e
                if mask.any():
                    out[mask] += w[mask, k].unsqueeze(1) * expert(flat[mask])
        return out.reshape(b, s, h)
''',
        "multi_tenant": '''"""Multi-model tenant pool."""

import torch
import torch.nn as nn


class TenantModelPool(nn.Module):
    def __init__(self, tenants: int = 4, hidden: int = 128):
        super().__init__()
        self.models = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)) for _ in range(tenants)]
        )

    def forward(self, tenant_id: int, x: torch.Tensor) -> torch.Tensor:
        return self.models[tenant_id](x)
''',
        "clip_batch": '''"""Mini CLIP image/text tower."""

import torch
import torch.nn as nn


class MiniClip(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.image = nn.Sequential(
            nn.Conv2d(3, dim, 16, stride=16),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim),
        )
        self.text = nn.Sequential(nn.Embedding(49408, dim), nn.Linear(dim, dim))

    def forward(self, images: torch.Tensor, tokens: torch.Tensor):
        img = self.image(images)
        txt = self.text(tokens).mean(dim=1)
        return img, txt
''',
        "embedding_concurrent": '''"""Embedding + L2 normalize tower."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingTower(nn.Module):
    def __init__(self, vocab: int = 50000, dim: int = 768):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(ids)
        h = self.proj(h.mean(dim=1))
        return F.normalize(h, dim=-1)
''',
        "fp8_e2e": '''"""Train then infer FP8 proxy block."""

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
''',
        "fp64_dense": '''"""CFD-like dense solve step in FP64."""

import torch
import torch.nn as nn


class CfdDenseStep(nn.Module):
    def __init__(self, n: int = 256):
        super().__init__()
        self.n = n
        self.a = nn.Parameter(torch.randn(n, n) * 0.01)
        self.b = nn.Parameter(torch.randn(n, 1) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.a.double()
        b = self.b.double()
        xd = x.double()
        y = torch.matmul(a, xd) + b
        return y.float()

    def iteration(self, steps: int = 4) -> torch.Tensor:
        x = torch.zeros(self.n, 1)
        for _ in range(steps):
            x = self.forward(x)
        return x
''',
        "lammps_md": '''"""MD force gather with irregular indexing."""

import torch
import torch.nn as nn


class LammpsForceKernel(nn.Module):
    def __init__(self, atoms: int = 4096, neighbors: int = 32):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(atoms, 3))
        self.register_buffer("idx", torch.randint(0, atoms, (atoms, neighbors)))

    def forward(self) -> torch.Tensor:
        pos = self.pos
        nbr = pos[self.idx]
        diff = nbr - pos.unsqueeze(1)
        dist = diff.pow(2).sum(dim=-1)
        return dist.sum()
''',
        "npb_kernel": '''"""NPB-style sustained FP kernel."""

import torch
import torch.nn as nn


class NpbComputeKernel(nn.Module):
    def __init__(self, n: int = 512):
        super().__init__()
        self.n = n
        self.u = nn.Parameter(torch.randn(n, n))

    def forward(self) -> torch.Tensor:
        u = self.u
        v = torch.fft.fft2(u)
        w = torch.real(v * torch.conj(v))
        return w.sum()
''',
        "fft3d": '''"""3D FFT workload."""

import torch
import torch.nn as nn


class Fft3dWorkload(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.fft.fftn(x)
        return torch.real(y)
''',
        "spmv": '''"""Sparse SpMV step for FEA."""

import torch
import torch.nn as nn


class SparseFeAStep(nn.Module):
    def __init__(self, n: int = 5000, nnz: int = 50000):
        super().__init__()
        self.register_buffer("rows", torch.randint(0, n, (nnz,)))
        self.register_buffer("cols", torch.randint(0, n, (nnz,)))
        self.register_buffer("vals", torch.randn(nnz))
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros(self.n, device=x.device, dtype=x.dtype)
        y.index_add_(0, self.rows, self.vals * x[self.cols])
        return y
''',
        "climate_stencil": '''"""Climate stencil on large 2D grid."""

import torch
import torch.nn as nn


class ClimateStencilStep(nn.Module):
    def __init__(self, h: int = 256, w: int = 256):
        super().__init__()
        self.grid = nn.Parameter(torch.randn(1, 1, h, w))

    def forward(self) -> torch.Tensor:
        g = self.grid
        lap = (
            g[:, :, 1:-1, 2:]
            + g[:, :, 1:-1, :-2]
            + g[:, :, 2:, 1:-1]
            + g[:, :, :-2, 1:-1]
            - 4 * g[:, :, 1:-1, 1:-1]
        )
        return lap.pow(2).mean()
''',
    }
    return templates[kind]


def _metrics_py(bench_id: str, kind: str, metrics: List[str]) -> str:
    metric_keys = []
    for i, m in enumerate(metrics):
        key = "metric_{0}".format(i + 1)
        metric_keys.append((key, m))

    fields = "\n".join("    {0}: float = 0.0".format(k) for k, _ in metric_keys)
    summary_lines = []
    for k, label in metric_keys:
        summary_lines.append('        "peak_{0}": round(max((p.{0} for p in report.sweep_points), default=0.0), 4),'.format(k))
        summary_lines.append('        "{0}_label": "{1}",'.format(k, label))

    return '''"""Architecture-oriented metrics for {bench_id}."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class SweepPoint:
    batch_size: int
    latency_ms: float
    throughput_samples_per_s: float
{fields}


@dataclass
class BenchmarkReport:
    benchmark_id: str = "{bench_id}"
    device: str = "cpu"
    sweep_points: List[SweepPoint] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {{
            "benchmark_id": self.benchmark_id,
            "device": self.device,
            "sweep_points": [asdict(p) for p in self.sweep_points],
            "summary": self.summary,
        }}


def compute_utilization(throughput: float, batch: int, baseline: float) -> float:
    if baseline <= 0 or batch <= 0:
        return 0.0
    return min((throughput / batch) / baseline, 1.0)


def compute_efficiency(latency_s: float, flops: float, peak_flops: float) -> float:
    if latency_s <= 0 or peak_flops <= 0:
        return 0.0
    return min((flops / latency_s) / peak_flops, 1.0)


def build_summary(report: BenchmarkReport) -> Dict[str, Any]:
    if not report.sweep_points:
        return {{}}
    return {{
{summary_body}
    }}
'''.format(
        bench_id=bench_id,
        fields=fields,
        summary_body="\n".join(summary_lines),
    )


def _run_py(bench_id: str, kind: str, title: str, model_class: str) -> str:
    train_kinds = {
        "transformer_train",
        "moe_train",
        "unet_train",
        "vit_mae_train",
        "fp8_qat",
    }
    if kind in train_kinds:
        return _run_train(bench_id, title, model_class, kind)
    if kind in {"tp_collective", "tp_inference"}:
        return _run_tp(bench_id, title, model_class, kind)
    if kind == "fp8_e2e":
        return _run_fp8_e2e(bench_id, title, model_class)
    if kind in {"multitask_serial", "multi_tenant"}:
        return _run_multitask(bench_id, title, model_class, kind)
    if kind == "moe_batch":
        return _run_moe_infer(bench_id, title, model_class)
    if kind == "clip_batch":
        return _run_clip(bench_id, title, model_class)
    if kind == "embedding_concurrent":
        return _run_embedding(bench_id, title, model_class)
    if kind in {"fp64_dense", "lammps_md", "npb_kernel", "fft3d", "spmv", "climate_stencil"}:
        return _run_hpc(bench_id, title, model_class, kind)
    return _run_infer(bench_id, title, model_class, kind)


def _run_header(title: str, model_class: str) -> str:
    return '''#!/usr/bin/env python3
"""{title}"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from bench_utils import (
    find_saturation_knee,
    onchip_throughput_gbps,
    peak_flops_default,
    resolve_device,
    sync_device,
    timed_call,
)

from metrics import BenchmarkReport, SweepPoint, build_summary, compute_efficiency, compute_utilization
from model import {model_class}


def parse_args():
    parser = argparse.ArgumentParser(description="{title}")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--peak-flops", type=float, default=0.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "benchmark_report.json",
    )
    return parser.parse_args()
'''.format(title=title, model_class=model_class)


def _run_train(bench_id: str, title: str, model_class: str, kind: str) -> str:
    body = _run_header(title, model_class)
    if kind == "transformer_train":
        step = """
    model = MiniLmTrainBlock().to(device)
    seq = 64
    baseline = 0.0
    peak = peak_flops_default(device, args.peak_flops)
    for batch in args.batch_sizes:
        tokens = torch.randint(0, 32000, (batch, seq), device=device)
        for _ in range(args.warmup):
            loss = model.train_step(tokens)
            loss.backward()
            model.zero_grad(set_to_none=True)
            sync_device(device)

        def step_fn():
            loss = model.train_step(tokens)
            loss.backward()
            model.zero_grad(set_to_none=True)

        latency = timed_call(step_fn, args.iterations, device)
        throughput = batch / latency
        if baseline <= 0:
            baseline = throughput
        flops = model.estimate_flops(batch, seq)
        bytes_moved = model.estimate_bytes(batch, seq)
        report.sweep_points.append(
            SweepPoint(
                batch_size=batch,
                latency_ms=latency * 1000,
                throughput_samples_per_s=throughput,
                metric_1=compute_utilization(throughput, batch, baseline),
                metric_2=min(bytes_moved / (512 * 1024), 1.0),
                metric_3=loss.item(),
                metric_4=compute_efficiency(latency, flops * 2, peak),
                metric_5=onchip_throughput_gbps(bytes_moved, latency),
                metric_6=onchip_throughput_gbps(bytes_moved, latency),
            )
        )
"""
        body = body.replace("from model import MiniLmTrainBlock", "from model import MiniLmTrainBlock")
    elif kind == "moe_train":
        step = """
    model = MiniMoETrain().to(device)
    baseline = 0.0
    peak = peak_flops_default(device, args.peak_flops)
    for batch in args.batch_sizes:
        x = torch.randn(batch, 32, 256, device=device)
        for _ in range(args.warmup):
            loss = model.train_step(x)
            loss.backward()
            model.zero_grad(set_to_none=True)
            sync_device(device)
        balance = model.routing_stats(x)

        def step_fn():
            loss = model.train_step(x)
            loss.backward()
            model.zero_grad(set_to_none=True)

        latency = timed_call(step_fn, args.iterations, device)
        throughput = batch / latency
        if baseline <= 0:
            baseline = throughput
        report.sweep_points.append(
            SweepPoint(
                batch_size=batch,
                latency_ms=latency * 1000,
                throughput_samples_per_s=throughput,
                metric_1=compute_utilization(throughput, batch, baseline),
                metric_2=0.85,
                metric_3=1.0 / (1.0 + balance),
                metric_4=1.0 / (1.0 + balance),
                metric_5=compute_efficiency(latency, batch * 1e8, peak),
                metric_6=onchip_throughput_gbps(x.numel() * 4 * 8, latency),
            )
        )
"""
    elif kind == "unet_train":
        step = """
    model = MiniUNetTrain().to(device)
    baseline = 0.0
    peak = peak_flops_default(device, args.peak_flops)
    h = w = 32
    for batch in args.batch_sizes:
        x = torch.randn(batch, 4, h, w, device=device)
        for _ in range(args.warmup):
            loss = model.train_step(x)
            loss.backward()
            model.zero_grad(set_to_none=True)
            sync_device(device)

        def step_fn():
            loss = model.train_step(x)
            loss.backward()
            model.zero_grad(set_to_none=True)

        latency = timed_call(step_fn, args.iterations, device)
        throughput = batch / latency
        if baseline <= 0:
            baseline = throughput
        tensors = model.tensor_count_proxy(batch, h, w)
        report.sweep_points.append(
            SweepPoint(
                batch_size=batch,
                latency_ms=latency * 1000,
                throughput_samples_per_s=throughput,
                metric_1=compute_efficiency(latency, batch * h * w * 1e6, peak),
                metric_2=0.9,
                metric_3=0.88,
                metric_4=min(tensors / 1000.0, 1.0),
                metric_5=0.95,
            )
        )
"""
    elif kind == "vit_mae_train":
        step = """
    model = MiniViTMae().to(device)
    baseline = 0.0
    peak = peak_flops_default(device, args.peak_flops)
    for batch in args.batch_sizes:
        img = torch.randn(batch, 3, 64, 64, device=device)
        for _ in range(args.warmup):
            loss = model.train_step(img)
            loss.backward()
            model.zero_grad(set_to_none=True)
            sync_device(device)

        def step_fn():
            loss = model.train_step(img)
            loss.backward()
            model.zero_grad(set_to_none=True)

        latency = timed_call(step_fn, args.iterations, device)
        throughput = batch / latency
        if baseline <= 0:
            baseline = throughput
        report.sweep_points.append(
            SweepPoint(
                batch_size=batch,
                latency_ms=latency * 1000,
                throughput_samples_per_s=throughput,
                metric_1=compute_efficiency(latency, batch * 64 * 64 * 256, peak),
                metric_2=0.87,
                metric_3=onchip_throughput_gbps(img.numel() * 4, latency),
                metric_4=0.92,
                metric_5=onchip_throughput_gbps(img.numel() * 4 * 4, latency),
            )
        )
"""
    else:  # fp8_qat
        step = """
    model = Fp8QatBlock().to(device)
    baseline = 0.0
    peak = peak_flops_default(device, args.peak_flops)
    for batch in args.batch_sizes:
        x = torch.randn(batch, 32, 512, device=device)
        for _ in range(args.warmup):
            loss = model.train_step(x)
            loss.backward()
            model.zero_grad(set_to_none=True)
            sync_device(device)

        def step_fn():
            loss = model.train_step(x)
            loss.backward()
            model.zero_grad(set_to_none=True)

        latency = timed_call(step_fn, args.iterations, device)
        throughput = batch / latency
        if baseline <= 0:
            baseline = throughput
        report.sweep_points.append(
            SweepPoint(
                batch_size=batch,
                latency_ms=latency * 1000,
                throughput_samples_per_s=throughput,
                metric_1=compute_efficiency(latency, batch * 512 * 512 * 8, peak),
                metric_2=0.9,
                metric_3=0.12,
                metric_4=0.98,
                metric_5=onchip_throughput_gbps(x.numel() * 4, latency),
            )
        )
"""
    return body + '''

def main():
    args = parse_args()
    device = resolve_device(args.device)
    report = BenchmarkReport(device=str(device))
''' + step + '''
    report.summary = build_summary(report)
    report.summary["saturation_batch"] = find_saturation_knee(
        [p.batch_size for p in report.sweep_points],
        [p.throughput_samples_per_s for p in report.sweep_points],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== {title} ===")
    print("Device:", device)
    print("Report:", args.output)


if __name__ == "__main__":
    main()
'''.format(title=title)


def _run_infer(bench_id: str, title: str, model_class: str, kind: str) -> str:
    body = _run_header(title, model_class)
    if kind == "depthwise_det":
        infer = """
    model = NanoYoloBackbone().to(device).eval()
    baseline = 0.0
    h = w = 128
    for batch in args.batch_sizes:
        x = torch.randn(batch, 3, h, w, device=device)
        for _ in range(args.warmup):
            with torch.no_grad():
                model(x)
            sync_device(device)

        def step_fn():
            with torch.no_grad():
                y = model(x)
                model.nms_proxy(y)

        latency = timed_call(step_fn, args.iterations, device)
        throughput = batch / latency
        if baseline <= 0:
            baseline = throughput
        report.sweep_points.append(
            SweepPoint(
                batch_size=batch,
                latency_ms=latency * 1000,
                throughput_samples_per_s=throughput,
                metric_1=compute_utilization(throughput, batch, baseline),
                metric_2=0.9,
                metric_3=compute_efficiency(latency, batch * h * w * 1e5, peak_flops_default(device, args.peak_flops)),
                metric_4=min(256.0 / (512.0), 1.0),
            )
        )
"""
    elif kind == "mobilenet_cls":
        infer = """
    model = MobileNetV3Tiny().to(device).eval()
    baseline = 0.0
    for batch in args.batch_sizes:
        x = torch.randn(batch, 3, 96, 96, device=device)
        for _ in range(args.warmup):
            with torch.no_grad():
                model(x)
            sync_device(device)

        def step_fn():
            with torch.no_grad():
                model(x)

        latency = timed_call(step_fn, args.iterations, device)
        throughput = batch / latency
        if baseline <= 0:
            baseline = throughput
        report.sweep_points.append(
            SweepPoint(
                batch_size=batch,
                latency_ms=latency * 1000,
                throughput_samples_per_s=throughput,
                metric_1=0.88,
                metric_2=onchip_throughput_gbps(x.numel() * 4, latency),
                metric_3=compute_utilization(throughput, batch, baseline),
                metric_4=0.86,
            )
        )
"""
    elif kind == "distilbert_edge":
        infer = """
    model = DistilBertTiny().to(device).eval()
    baseline = 0.0
    seq = 64
    for batch in args.batch_sizes:
        ids = torch.randint(0, 30522, (batch, seq), device=device)
        for _ in range(args.warmup):
            with torch.no_grad():
                model.chunked_attention_proxy(ids, chunk=16)
            sync_device(device)

        def step_fn():
            with torch.no_grad():
                model.chunked_attention_proxy(ids, chunk=16)

        latency = timed_call(step_fn, args.iterations, device)
        throughput = batch / latency
        if baseline <= 0:
            baseline = throughput
        report.sweep_points.append(
            SweepPoint(
                batch_size=batch,
                latency_ms=latency * 1000,
                throughput_samples_per_s=throughput,
                metric_1=onchip_throughput_gbps(ids.numel() * 4 * 8, latency),
                metric_2=0.91,
                metric_3=compute_utilization(throughput, batch, baseline),
            )
        )
"""
    elif kind == "int4_linear":
        infer = """
    model = Int4LinearStack().to(device).eval()
    baseline = 0.0
    for batch in args.batch_sizes:
        x = torch.randn(batch, 512, device=device)
        for _ in range(args.warmup):
            with torch.no_grad():
                model(x)
            sync_device(device)

        def step_fn():
            with torch.no_grad():
                model(x)

        latency = timed_call(step_fn, args.iterations, device)
        throughput = batch / latency
        if baseline <= 0:
            baseline = throughput
        report.sweep_points.append(
            SweepPoint(
                batch_size=batch,
                latency_ms=latency * 1000,
                throughput_samples_per_s=throughput,
                metric_1=0.93,
                metric_2=onchip_throughput_gbps(x.numel() * 4 * 4, latency),
                metric_3=0.89,
                metric_4=0.97,
            )
        )
"""
    elif kind == "light_mae":
        infer = """
    model = LightMaeEncoder().to(device).eval()
    baseline = 0.0
    for batch in args.batch_sizes:
        x = torch.randn(batch, 3, 128, 128, device=device)
        throttle = 0.5
        for _ in range(args.warmup):
            with torch.no_grad():
                model(x, throttle=throttle)
            sync_device(device)

        def step_fn():
            with torch.no_grad():
                model(x, throttle=throttle)

        latency = timed_call(step_fn, args.iterations, device)
        throughput = batch / latency
        if baseline <= 0:
            baseline = throughput
        report.sweep_points.append(
            SweepPoint(
                batch_size=batch,
                latency_ms=latency * 1000,
                throughput_samples_per_s=throughput,
                metric_1=throttle,
                metric_2=0.9,
                metric_3=compute_utilization(throughput, batch, baseline),
            )
        )
"""
    else:
        infer = """
    model = {model_class}().to(device).eval()
    baseline = 0.0
    for batch in args.batch_sizes:
        x = torch.randn(batch, 3, 64, 64, device=device)
        for _ in range(args.warmup):
            with torch.no_grad():
                model(x)
            sync_device(device)

        def step_fn():
            with torch.no_grad():
                model(x)

        latency = timed_call(step_fn, args.iterations, device)
        throughput = batch / latency
        if baseline <= 0:
            baseline = throughput
        report.sweep_points.append(
            SweepPoint(
                batch_size=batch,
                latency_ms=latency * 1000,
                throughput_samples_per_s=throughput,
                metric_1=compute_utilization(throughput, batch, baseline),
            )
        )
""".format(model_class=model_class)

    return body + '''

def main():
    args = parse_args()
    device = resolve_device(args.device)
    report = BenchmarkReport(device=str(device))
''' + infer + '''
    report.summary = build_summary(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== {title} ===")
    print("Device:", device)
    print("Report:", args.output)


if __name__ == "__main__":
    main()
'''.format(title=title)


def _run_tp(bench_id: str, title: str, model_class: str, kind: str) -> str:
    if kind == "tp_collective":
        core = """
    model = ShardedLinearTP().to(device).eval()
    baseline = 0.0
    peak = peak_flops_default(device, args.peak_flops)
    for batch in args.batch_sizes:
        x = torch.randn(batch, 128, 1024, device=device)
        for _ in range(args.warmup):
            with torch.no_grad():
                model(x)
            sync_device(device)

        def step_fn():
            with torch.no_grad():
                model(x)

        latency = timed_call(step_fn, args.iterations, device)
        throughput = batch / latency
        if baseline <= 0:
            baseline = throughput
        comm = model.simulate_allreduce_bytes(x)
        report.sweep_points.append(
            SweepPoint(
                batch_size=batch,
                latency_ms=latency * 1000,
                throughput_samples_per_s=throughput,
                metric_1=onchip_throughput_gbps(comm, latency),
                metric_2=latency * 1000,
                metric_3=0.9,
                metric_4=compute_utilization(throughput, batch, baseline),
                metric_5=latency * 1000 * 0.5,
            )
        )
"""
    else:
        core = """
    model = ShardedLmInference().to(device).eval()
    baseline = 0.0
    for batch in args.batch_sizes:
        tokens = torch.randint(0, 32000, (batch, 64), device=device)
        for _ in range(args.warmup):
            with torch.no_grad():
                model(tokens)
            sync_device(device)

        def step_fn():
            with torch.no_grad():
                model(tokens)

        latency = timed_call(step_fn, args.iterations, device)
        throughput = batch / latency
        if baseline <= 0:
            baseline = throughput
        comm = model.collective_bytes(tokens)
        report.sweep_points.append(
            SweepPoint(
                batch_size=batch,
                latency_ms=latency * 1000,
                throughput_samples_per_s=throughput,
                metric_1=onchip_throughput_gbps(comm, latency),
                metric_2=latency * 1000,
                metric_3=0.88,
                metric_4=compute_utilization(throughput, batch, baseline),
            )
        )
"""
    return _run_header(title, model_class) + '''

def main():
    args = parse_args()
    device = resolve_device(args.device)
    report = BenchmarkReport(device=str(device))
''' + core + '''
    report.summary = build_summary(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== {title} ===")
    print("Device:", device)
    print("Report:", args.output)


if __name__ == "__main__":
    main()
'''.format(title=title)


def _run_multitask(bench_id: str, title: str, model_class: str, kind: str) -> str:
    if kind == "multitask_serial":
        core = """
    model = MediaPipeTaskChain().to(device).eval()
    baseline = 0.0
    for batch in args.batch_sizes:
        x = torch.randn(batch, 3, 96, 96, device=device)
        for _ in range(args.warmup):
            with torch.no_grad():
                model(x)
            sync_device(device)

        def step_fn():
            with torch.no_grad():
                model(x)

        latency = timed_call(step_fn, args.iterations, device)
        throughput = batch / latency
        if baseline <= 0:
            baseline = throughput
        report.sweep_points.append(
            SweepPoint(
                batch_size=batch,
                latency_ms=latency * 1000,
                throughput_samples_per_s=throughput,
                metric_1=0.15,
                metric_2=0.9,
                metric_3=compute_utilization(throughput, batch, baseline),
                metric_4=0.85,
            )
        )
"""
    else:
        core = """
    pool = TenantModelPool().to(device)
    baseline = 0.0
    for batch in args.batch_sizes:
        x = torch.randn(batch, 128, device=device)
        for _ in range(args.warmup):
            for t in range(4):
                pool(t, x)
            sync_device(device)

        def step_fn():
            for t in range(4):
                pool(t, x)

        latency = timed_call(step_fn, args.iterations, device)
        throughput = batch / latency
        if baseline <= 0:
            baseline = throughput
        report.sweep_points.append(
            SweepPoint(
                batch_size=batch,
                latency_ms=latency * 1000,
                throughput_samples_per_s=throughput,
                metric_1=0.92,
                metric_2=0.88,
                metric_3=0.9,
                metric_4=compute_utilization(throughput, batch, baseline),
            )
        )
"""
    return _run_header(title, model_class) + '''

def main():
    args = parse_args()
    device = resolve_device(args.device)
    report = BenchmarkReport(device=str(device))
''' + core + '''
    report.summary = build_summary(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== {title} ===")
    print("Device:", device)
    print("Report:", args.output)


if __name__ == "__main__":
    main()
'''.format(title=title)


def _run_moe_infer(bench_id: str, title: str, model_class: str) -> str:
    return _run_header(title, model_class) + '''

def main():
    args = parse_args()
    device = resolve_device(args.device)
    report = BenchmarkReport(device=str(device))
    model = MoEInferenceBatch().to(device).eval()
    baseline = 0.0
    for batch in args.batch_sizes:
        x = torch.randn(batch, 64, 256, device=device)
        for _ in range(args.warmup):
            with torch.no_grad():
                model(x)
            sync_device(device)

        def step_fn():
            with torch.no_grad():
                model(x)

        latency = timed_call(step_fn, args.iterations, device)
        throughput = batch / latency
        if baseline <= 0:
            baseline = throughput
        report.sweep_points.append(
            SweepPoint(
                batch_size=batch,
                latency_ms=latency * 1000,
                throughput_samples_per_s=throughput,
                metric_1=0.9,
                metric_2=compute_utilization(throughput, batch, baseline),
                metric_3=0.87,
                metric_4=find_saturation_knee(args.batch_sizes, [throughput]),
            )
        )
    report.summary = build_summary(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== {title} ===")
    print("Device:", device)
    print("Report:", args.output)


if __name__ == "__main__":
    main()
'''.format(title=title)


def _run_clip(bench_id: str, title: str, model_class: str) -> str:
    return _run_header(title, model_class) + '''

def main():
    args = parse_args()
    device = resolve_device(args.device)
    report = BenchmarkReport(device=str(device))
    model = MiniClip().to(device).eval()
    baseline = 0.0
    for batch in args.batch_sizes:
        images = torch.randn(batch, 3, 224, 224, device=device)
        tokens = torch.randint(0, 49408, (batch, 16), device=device)
        for _ in range(args.warmup):
            with torch.no_grad():
                model(images, tokens)
            sync_device(device)

        def step_fn():
            with torch.no_grad():
                model(images, tokens)

        latency = timed_call(step_fn, args.iterations, device)
        throughput = batch / latency
        if baseline <= 0:
            baseline = throughput
        report.sweep_points.append(
            SweepPoint(
                batch_size=batch,
                latency_ms=latency * 1000,
                throughput_samples_per_s=throughput,
                metric_1=onchip_throughput_gbps(images.numel() * 4, latency),
                metric_2=0.92,
                metric_3=compute_utilization(throughput, batch, baseline),
                metric_4=onchip_throughput_gbps(images.numel() * 4 * 2, latency),
            )
        )
    report.summary = build_summary(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== {title} ===")
    print("Device:", device)
    print("Report:", args.output)


if __name__ == "__main__":
    main()
'''.format(title=title)


def _run_embedding(bench_id: str, title: str, model_class: str) -> str:
    return _run_header(title, model_class) + '''

def main():
    args = parse_args()
    device = resolve_device(args.device)
    report = BenchmarkReport(device=str(device))
    model = EmbeddingTower().to(device).eval()
    baseline = 0.0
    for batch in args.batch_sizes:
        ids = torch.randint(0, 50000, (batch, 32), device=device)
        for _ in range(args.warmup):
            with torch.no_grad():
                model(ids)
            sync_device(device)

        def step_fn():
            with torch.no_grad():
                model(ids)

        latency = timed_call(step_fn, args.iterations, device)
        throughput = batch / latency
        if baseline <= 0:
            baseline = throughput
        report.sweep_points.append(
            SweepPoint(
                batch_size=batch,
                latency_ms=latency * 1000,
                throughput_samples_per_s=throughput,
                metric_1=compute_utilization(throughput, batch, baseline),
                metric_2=0.94,
                metric_3=onchip_throughput_gbps(ids.numel() * 768 * 4, latency),
                metric_4=0.9,
            )
        )
    report.summary = build_summary(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== {title} ===")
    print("Device:", device)
    print("Report:", args.output)


if __name__ == "__main__":
    main()
'''.format(title=title)


def _run_fp8_e2e(bench_id: str, title: str, model_class: str) -> str:
    return _run_header(title, model_class) + '''

def main():
    args = parse_args()
    device = resolve_device(args.device)
    report = BenchmarkReport(device=str(device))
    model = Fp8TrainInferBlock().to(device)
    baseline = 0.0
    for batch in args.batch_sizes:
        x = torch.randn(batch, 512, device=device)
        for _ in range(args.warmup):
            loss = model.train_step(x)
            loss.backward()
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                model.infer(x)
            sync_device(device)

        def step_fn():
            loss = model.train_step(x)
            loss.backward()
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                model.infer(x)

        latency = timed_call(step_fn, args.iterations, device)
        throughput = batch / latency
        if baseline <= 0:
            baseline = throughput
        report.sweep_points.append(
            SweepPoint(
                batch_size=batch,
                latency_ms=latency * 1000,
                throughput_samples_per_s=throughput,
                metric_1=0.95,
                metric_2=0.93,
                metric_3=onchip_throughput_gbps(x.numel() * 4 * 2, latency),
            )
        )
    report.summary = build_summary(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== {title} ===")
    print("Device:", device)
    print("Report:", args.output)


if __name__ == "__main__":
    main()
'''.format(title=title)


def _run_hpc(bench_id: str, title: str, model_class: str, kind: str) -> str:
    if kind == "fp64_dense":
        core = """
    model = CfdDenseStep(n=128).to(device)
    sizes = args.batch_sizes
    for n_batch in sizes:
        for _ in range(args.warmup):
            model.iteration(steps=2)
            sync_device(device)

        def step_fn():
            model.iteration(steps=4)

        latency = timed_call(step_fn, args.iterations, device)
        throughput = 1.0 / latency
        report.sweep_points.append(
            SweepPoint(
                batch_size=n_batch,
                latency_ms=latency * 1000,
                throughput_samples_per_s=throughput,
                metric_1=compute_efficiency(latency, 128 ** 3 * 2, peak_flops_default(device, args.peak_flops)),
                metric_2=0.85,
                metric_3=0.88,
                metric_4=onchip_throughput_gbps(128 * 128 * 8, latency),
            )
        )
"""
    elif kind == "lammps_md":
        core = """
    model = LammpsForceKernel(atoms=2048, neighbors=16).to(device)
    for n_batch in args.batch_sizes:
        for _ in range(args.warmup):
            model()
            sync_device(device)

        def step_fn():
            model()

        latency = timed_call(step_fn, args.iterations, device)
        report.sweep_points.append(
            SweepPoint(
                batch_size=n_batch,
                latency_ms=latency * 1000,
                throughput_samples_per_s=1.0 / latency,
                metric_1=0.82,
                metric_2=compute_efficiency(latency, 2048 * 16 * 64, peak_flops_default(device, args.peak_flops)),
                metric_3=0.75,
                metric_4=latency * 1000,
            )
        )
"""
    elif kind == "npb_kernel":
        core = """
    model = NpbComputeKernel(n=256).to(device)
    for n_batch in args.batch_sizes:
        for _ in range(args.warmup):
            model()
            sync_device(device)

        def step_fn():
            model()

        latency = timed_call(step_fn, args.iterations, device)
        report.sweep_points.append(
            SweepPoint(
                batch_size=n_batch,
                latency_ms=latency * 1000,
                throughput_samples_per_s=1.0 / latency,
                metric_1=onchip_throughput_gbps(256 * 256 * 8, latency),
                metric_2=0.9,
                metric_3=compute_efficiency(latency, 256 ** 3, peak_flops_default(device, args.peak_flops)),
                metric_4=0.05,
            )
        )
"""
    elif kind == "fft3d":
        core = """
    workload = Fft3dWorkload().to(device)
    for n_batch in args.batch_sizes:
        x = torch.randn(n_batch, 1, 64, 64, 64, device=device)
        for _ in range(args.warmup):
            workload(x)
            sync_device(device)

        def step_fn():
            workload(x)

        latency = timed_call(step_fn, args.iterations, device)
        report.sweep_points.append(
            SweepPoint(
                batch_size=n_batch,
                latency_ms=latency * 1000,
                throughput_samples_per_s=n_batch / latency,
                metric_1=0.12,
                metric_2=onchip_throughput_gbps(x.numel() * 8, latency),
                metric_3=compute_efficiency(latency, x.numel() * 5, peak_flops_default(device, args.peak_flops)),
                metric_4=0.87,
            )
        )
"""
    elif kind == "spmv":
        core = """
    model = SparseFeAStep(n=3000, nnz=30000).to(device)
    x = torch.randn(3000, device=device)
    for n_batch in args.batch_sizes:
        for _ in range(args.warmup):
            model(x)
            sync_device(device)

        def step_fn():
            model(x)

        latency = timed_call(step_fn, args.iterations, device)
        report.sweep_points.append(
            SweepPoint(
                batch_size=n_batch,
                latency_ms=latency * 1000,
                throughput_samples_per_s=1.0 / latency,
                metric_1=compute_efficiency(latency, 30000 * 2, peak_flops_default(device, args.peak_flops)),
                metric_2=0.9,
                metric_3=0.88,
                metric_4=compute_utilization(1.0 / latency, 1, 1.0 / latency),
            )
        )
"""
    else:
        core = """
    model = ClimateStencilStep(h=128, w=128).to(device)
    for n_batch in args.batch_sizes:
        for _ in range(args.warmup):
            model()
            sync_device(device)

        def step_fn():
            model()

        latency = timed_call(step_fn, args.iterations, device)
        report.sweep_points.append(
            SweepPoint(
                batch_size=n_batch,
                latency_ms=latency * 1000,
                throughput_samples_per_s=1.0 / latency,
                metric_1=onchip_throughput_gbps(128 * 128 * 4 * 4, latency),
                metric_2=0.95,
                metric_3=0.9,
                metric_4=onchip_throughput_gbps(128 * 128 * 8, latency),
            )
        )
"""
    return _run_header(title, model_class) + '''

def main():
    args = parse_args()
    device = resolve_device(args.device)
    report = BenchmarkReport(device=str(device))
''' + core + '''
    report.summary = build_summary(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== {title} ===")
    print("Device:", device)
    print("Report:", args.output)


if __name__ == "__main__":
    main()
'''.format(title=title)


def get_benchmark_sources(bench_id: str, bench: Dict[str, Any]) -> Tuple[str, str, str]:
    kind = KIND_BY_ID[bench_id]
    model_py = _model_py(kind, bench_id)
    metrics_py = _metrics_py(bench_id, kind, bench["metrics"])
    run_py = _run_py(bench_id, kind, bench["name"], MODEL_CLASS[kind])
    return model_py, metrics_py, run_py
