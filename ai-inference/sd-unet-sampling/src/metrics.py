from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class SweepPoint:
    batch_size: int
    sampling_steps: int
    loop_launch_overhead_ratio: float
    conv_array_utilization: float
    intermediate_tensor_cache_reuse_gbps: float
    latency_per_step_ms: float


@dataclass
class BenchmarkReport:
    benchmark_id: str = "sd-unet-sampling"
    device: str = "cpu"
    sweep_points: List[SweepPoint] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "device": self.device,
            "sweep_points": [asdict(p) for p in self.sweep_points],
            "summary": self.summary,
        }


def build_summary(report: BenchmarkReport) -> Dict[str, Any]:
    if not report.sweep_points:
        return {}
    return {
        "min_loop_launch_overhead_ratio": round(min(p.loop_launch_overhead_ratio for p in report.sweep_points), 4),
        "peak_conv_array_utilization": round(max(p.conv_array_utilization for p in report.sweep_points), 4),
        "peak_intermediate_tensor_cache_reuse_gbps": round(max(p.intermediate_tensor_cache_reuse_gbps for p in report.sweep_points), 4),
        "min_latency_per_step_ms": round(min(p.latency_per_step_ms for p in report.sweep_points), 4),
    }
