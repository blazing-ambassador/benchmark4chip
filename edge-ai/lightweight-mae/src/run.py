#!/usr/bin/env python3
"""轻量化 MAE 视觉提取推理"""

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
from model import LightMaeEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="轻量化 MAE 视觉提取推理")
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

    report.summary = build_summary(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== 轻量化 MAE 视觉提取推理 ===")
    print("Device:", device)
    print("Report:", args.output)


if __name__ == "__main__":
    main()
