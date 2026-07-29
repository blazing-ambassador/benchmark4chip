from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class SweepPoint:
    batch_size: int
    precision: str
    throughput_samples_per_s: float
    quant_dequant_overhead_ratio: float
    zero_point_offset_ops_ratio: float
    low_precision_error_tolerance: float


@dataclass
class BenchmarkReport:
    benchmark_id: str = "int8-fp8-quant-inference"
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
    int8 = [p for p in report.sweep_points if p.precision == "int8"]
    fp8 = [p for p in report.sweep_points if p.precision == "fp8"]
    return {
        "peak_int8_throughput_samples_per_s": round(max((p.throughput_samples_per_s for p in int8), default=0.0), 2),
        "peak_fp8_throughput_samples_per_s": round(max((p.throughput_samples_per_s for p in fp8), default=0.0), 2),
        "min_quant_dequant_overhead_ratio": round(min(p.quant_dequant_overhead_ratio for p in report.sweep_points), 4),
        "peak_zero_point_offset_ops_ratio": round(max(p.zero_point_offset_ops_ratio for p in report.sweep_points), 4),
        "min_low_precision_error_tolerance": round(min(p.low_precision_error_tolerance for p in report.sweep_points), 6),
    }
