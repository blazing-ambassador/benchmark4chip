from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class SweepPoint:
    batch_size: int
    pipeline_parallel_efficiency: float
    model_switch_overhead_ms: float
    feature_projection_transfer_gbps: float
    multitask_isolation_overhead_ratio: float
    end_to_end_latency_ms: float


@dataclass
class BenchmarkReport:
    benchmark_id: str = "llava-multimodal"
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
        "peak_pipeline_parallel_efficiency": round(max(p.pipeline_parallel_efficiency for p in report.sweep_points), 4),
        "min_model_switch_overhead_ms": round(min(p.model_switch_overhead_ms for p in report.sweep_points), 4),
        "peak_feature_projection_transfer_gbps": round(max(p.feature_projection_transfer_gbps for p in report.sweep_points), 4),
        "min_multitask_isolation_overhead_ratio": round(min(p.multitask_isolation_overhead_ratio for p in report.sweep_points), 4),
    }
