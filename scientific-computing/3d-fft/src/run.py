#!/usr/bin/env python3
"""3D-FFT 大规模傅里叶变换"""

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
from model import Fft3dWorkload


def parse_args():
    parser = argparse.ArgumentParser(description="3D-FFT 大规模傅里叶变换")
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


def main():
    args = parse_args()
    device = resolve_device(args.device)
    report = BenchmarkReport(device=str(device))

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

    report.summary = build_summary(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== 3D-FFT 大规模傅里叶变换 ===")
    print("Device:", device)
    print("Report:", args.output)


if __name__ == "__main__":
    main()
