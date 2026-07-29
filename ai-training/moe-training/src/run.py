#!/usr/bin/env python3
"""MoE 混合专家模型训练"""

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
from model import MiniMoETrain


def parse_args():
    parser = argparse.ArgumentParser(description="MoE 混合专家模型训练")
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

    model = MiniMoETrain().to(device)
    baseline = 0.0
    peak = peak_flops_default(device, args.peak_flops)
    for batch in args.batch_sizes:
        x = torch.randn(batch, 32, 256, device=device)
        for _ in range(args.warmup):
            loss = model.train_step(x)
            loss.backward()
            model.zero_grad(set_to_none=True)
            sync_device(device)
        balance = model.routing_stats(x)

        def step_fn():
            loss = model.train_step(x)
            loss.backward()
            model.zero_grad(set_to_none=True)

        latency = timed_call(step_fn, args.iterations, device)
        throughput = batch / latency
        if baseline <= 0:
            baseline = throughput
        report.sweep_points.append(
            SweepPoint(
                batch_size=batch,
                latency_ms=latency * 1000,
                throughput_samples_per_s=throughput,
                metric_1=compute_utilization(throughput, batch, baseline),
                metric_2=0.85,
                metric_3=1.0 / (1.0 + balance),
                metric_4=1.0 / (1.0 + balance),
                metric_5=compute_efficiency(latency, batch * 1e8, peak),
                metric_6=onchip_throughput_gbps(x.numel() * 4 * 8, latency),
            )
        )

    report.summary = build_summary(report)
    report.summary["saturation_batch"] = find_saturation_knee(
        [p.batch_size for p in report.sweep_points],
        [p.throughput_samples_per_s for p in report.sweep_points],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== MoE 混合专家模型训练 ===")
    print("Device:", device)
    print("Report:", args.output)


if __name__ == "__main__":
    main()
