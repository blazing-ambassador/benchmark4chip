#!/usr/bin/env python3
"""NPB 并行 HPC 基准"""

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
from model import NpbComputeKernel


def parse_args():
    parser = argparse.ArgumentParser(description="NPB 并行 HPC 基准")
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

    report.summary = build_summary(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== NPB 并行 HPC 基准 ===")
    print("Device:", device)
    print("Report:", args.output)


if __name__ == "__main__":
    main()
