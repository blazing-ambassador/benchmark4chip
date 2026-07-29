"""Architecture-oriented metrics for BERT batch encoding benchmark."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class SweepPoint:
    batch_size: int
    seq_len: int
    latency_ms: float
    throughput_samples_per_s: float
    throughput_tokens_per_s: float
    batch_parallel_utilization: float
    short_seq_scheduling_efficiency: float
    onchip_cache_batch_throughput_gbps: float


@dataclass
class BenchmarkReport:
    benchmark_id: str = "bert-batch-encoding"
    device: str = "cpu"
    dtype: str = "float32"
    baseline_throughput_samples_per_s: float = 0.0
    compute_overflow_threshold_batch: int = 0
    compute_overflow_threshold_concurrency: int = 0
    sweep_points: List[SweepPoint] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "device": self.device,
            "dtype": self.dtype,
            "baseline_throughput_samples_per_s": self.baseline_throughput_samples_per_s,
            "compute_overflow_threshold_batch": self.compute_overflow_threshold_batch,
            "compute_overflow_threshold_concurrency": self.compute_overflow_threshold_concurrency,
            "sweep_points": [asdict(p) for p in self.sweep_points],
            "summary": self.summary,
        }


def compute_batch_parallel_utilization(
    throughput_at_batch: float,
    batch_size: int,
    baseline_throughput: float,
) -> float:
    """
    Parallel unit utilization proxy:
    (per-sample throughput at batch B) / (per-sample throughput at batch 1).
    """
    if baseline_throughput <= 0 or batch_size <= 0:
        return 0.0
    per_sample = throughput_at_batch / batch_size
    return min(per_sample / baseline_throughput, 1.0)


def compute_short_seq_scheduling_efficiency(
    latency_s: float,
    seq_len: int,
    batch_size: int,
    flops: float,
    peak_flops_per_s: float,
) -> float:
    """
    Scheduling efficiency proxy:
    achieved compute / ideal compute if units were fully busy.
    """
    if latency_s <= 0 or peak_flops_per_s <= 0:
        return 0.0
    achieved = flops / latency_s
    return min(achieved / peak_flops_per_s, 1.0)


def compute_onchip_cache_throughput_gbps(bytes_moved: int, latency_s: float) -> float:
    """On-chip cache batch data throughput proxy in GB/s."""
    if latency_s <= 0:
        return 0.0
    return bytes_moved / latency_s / 1e9


def find_overflow_threshold(
    batch_sizes: List[int],
    throughputs: List[float],
    growth_ratio: float = 0.05,
) -> int:
    """
    Batch size where marginal throughput gain drops below `growth_ratio`.
    Marks compute saturation / overflow knee point.
    """
    if not batch_sizes:
        return 0

    best_batch = batch_sizes[0]
    best_throughput = throughputs[0]

    for batch, throughput in zip(batch_sizes, throughputs):
        if throughput > best_throughput * (1.0 + growth_ratio):
            best_throughput = throughput
            best_batch = batch

    return best_batch


def find_concurrency_overflow_threshold(
    concurrencies: List[int],
    throughputs: List[float],
    growth_ratio: float = 0.05,
) -> int:
    """Concurrent stream count at throughput saturation."""
    return find_overflow_threshold(concurrencies, throughputs, growth_ratio)


def build_summary(report: BenchmarkReport) -> Dict[str, Any]:
    if not report.sweep_points:
        return {}

    best_util = max(p.batch_parallel_utilization for p in report.sweep_points)
    best_sched = max(p.short_seq_scheduling_efficiency for p in report.sweep_points)
    best_cache = max(p.onchip_cache_batch_throughput_gbps for p in report.sweep_points)
    peak_tokens = max(p.throughput_tokens_per_s for p in report.sweep_points)

    return {
        "peak_batch_parallel_utilization": round(best_util, 4),
        "peak_short_seq_scheduling_efficiency": round(best_sched, 4),
        "compute_overflow_threshold_batch": report.compute_overflow_threshold_batch,
        "compute_overflow_threshold_concurrency": report.compute_overflow_threshold_concurrency,
        "peak_onchip_cache_batch_throughput_gbps": round(best_cache, 4),
        "peak_throughput_tokens_per_s": round(peak_tokens, 2),
    }
