"""Metrics for LLaMA incremental decode benchmark."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class SweepPoint:
    batch_size: int
    context_len: int
    decode_steps: int
    time_to_first_token_ms: float
    per_token_decode_latency_ms: float
    kv_cache_residency_ratio: float
    seq_length_compute_elasticity: float
    batch_scheduling_utilization: float
    weight_bandwidth_bottleneck_gbps: float


@dataclass
class BenchmarkReport:
    benchmark_id: str = "llama-7b-decode"
    device: str = "cpu"
    dtype: str = "float32"
    simulated_onchip_cache_kb: int = 512
    sweep_points: List[SweepPoint] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "device": self.device,
            "dtype": self.dtype,
            "simulated_onchip_cache_kb": self.simulated_onchip_cache_kb,
            "sweep_points": [asdict(p) for p in self.sweep_points],
            "summary": self.summary,
        }


def kv_cache_residency_ratio(kv_bytes: int, cache_bytes: int) -> float:
    if kv_bytes <= 0:
        return 1.0
    return min(cache_bytes / kv_bytes, 1.0)


def seq_length_compute_elasticity(baseline_tps: float, current_tps: float) -> float:
    if baseline_tps <= 0:
        return 0.0
    return min(current_tps / baseline_tps, 1.0)


def batch_scheduling_utilization(per_sample_tps: float, baseline_per_sample_tps: float) -> float:
    if baseline_per_sample_tps <= 0:
        return 0.0
    return min(per_sample_tps / baseline_per_sample_tps, 1.0)


def weight_bandwidth_bottleneck_gbps(weight_bytes: int, latency_s: float) -> float:
    if latency_s <= 0:
        return 0.0
    return weight_bytes / latency_s / 1e9


def build_summary(report: BenchmarkReport) -> Dict[str, Any]:
    if not report.sweep_points:
        return {}
    return {
        "peak_kv_cache_residency_ratio": round(max(p.kv_cache_residency_ratio for p in report.sweep_points), 4),
        "peak_seq_length_compute_elasticity": round(max(p.seq_length_compute_elasticity for p in report.sweep_points), 4),
        "min_time_to_first_token_ms": round(min(p.time_to_first_token_ms for p in report.sweep_points), 4),
        "min_per_token_decode_latency_ms": round(min(p.per_token_decode_latency_ms for p in report.sweep_points), 4),
        "peak_batch_scheduling_utilization": round(max(p.batch_scheduling_utilization for p in report.sweep_points), 4),
        "peak_weight_bandwidth_bottleneck_gbps": round(max(p.weight_bandwidth_bottleneck_gbps for p in report.sweep_points), 4),
    }
