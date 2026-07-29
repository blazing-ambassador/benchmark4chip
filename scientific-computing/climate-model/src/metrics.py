"""Architecture-oriented metrics for climate-model."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class SweepPoint:
    batch_size: int
    latency_ms: float
    throughput_samples_per_s: float
    metric_1: float = 0.0
    metric_2: float = 0.0
    metric_3: float = 0.0
    metric_4: float = 0.0


@dataclass
class BenchmarkReport:
    benchmark_id: str = "climate-model"
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
        return {}
    return {
        "peak_metric_1": round(max((p.metric_1 for p in report.sweep_points), default=0.0), 4),
        "metric_1_label": "超大型数组流式读写带宽",
        "peak_metric_2": round(max((p.metric_2 for p in report.sweep_points), default=0.0), 4),
        "metric_2_label": "长时间步迭代算力稳定性",
        "peak_metric_3": round(max((p.metric_3 for p in report.sweep_points), default=0.0), 4),
        "metric_3_label": "循环变量寄存器复用",
        "peak_metric_4": round(max((p.metric_4 for p in report.sweep_points), default=0.0), 4),
        "metric_4_label": "大容量 DRAM 吞吐上限",
    }
