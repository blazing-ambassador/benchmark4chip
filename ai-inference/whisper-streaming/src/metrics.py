from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class SweepPoint:
    batch_size: int
    window_size: int
    total_frames: int
    sliding_window_segment_throughput_fps: float
    context_memory_growth_mb: float
    temporal_attention_latency_ms: float
    streaming_io_throughput_gbps: float


@dataclass
class BenchmarkReport:
    benchmark_id: str = "whisper-streaming"
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
        "peak_sliding_window_segment_throughput_fps": round(
            max(p.sliding_window_segment_throughput_fps for p in report.sweep_points), 4
        ),
        "max_context_memory_growth_mb": round(max(p.context_memory_growth_mb for p in report.sweep_points), 4),
        "min_temporal_attention_latency_ms": round(min(p.temporal_attention_latency_ms for p in report.sweep_points), 4),
        "peak_streaming_io_throughput_gbps": round(max(p.streaming_io_throughput_gbps for p in report.sweep_points), 4),
    }
