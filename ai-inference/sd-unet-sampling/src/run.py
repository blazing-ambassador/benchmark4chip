#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from bench_utils import onchip_throughput_gbps, peak_flops_default, resolve_device, sync_device, timed_call

from metrics import BenchmarkReport, SweepPoint, build_summary
from model import MiniSDUNet


def parse_args():
    p = argparse.ArgumentParser(description="SD UNet iterative sampling benchmark")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--steps", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--peak-flops", type=float, default=0.0)
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "results" / "benchmark_report.json")
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    peak_flops = peak_flops_default(device, args.peak_flops)
    model = MiniSDUNet().to(device).eval()
    report = BenchmarkReport(device=str(device))

    for steps in args.steps:
        for batch in args.batch_sizes:
            x = torch.randn(batch, 4, 32, 32, device=device)

            def one_step():
                with torch.no_grad():
                    model(x)

            for _ in range(args.warmup):
                one_step()
                sync_device(device)

            t_step = timed_call(one_step, args.iterations, device)

            def multi_step():
                cur = x
                with torch.no_grad():
                    for _ in range(steps):
                        cur = model(cur)

            for _ in range(args.warmup):
                multi_step()
                sync_device(device)

            t_loop = timed_call(multi_step, max(1, args.iterations // 2), device)

            ideal = t_step * steps
            overhead_ratio = max((t_loop - ideal) / max(t_loop, 1e-9), 0.0)
            macs = model.estimate_conv_macs(batch)
            achieved = macs * steps / t_loop
            conv_util = min(achieved / peak_flops, 1.0)
            inter_bytes = model.intermediate_bytes(batch)
            cache_gbps = onchip_throughput_gbps(inter_bytes * steps, t_loop)

            report.sweep_points.append(
                SweepPoint(
                    batch_size=batch,
                    sampling_steps=steps,
                    loop_launch_overhead_ratio=overhead_ratio,
                    conv_array_utilization=conv_util,
                    intermediate_tensor_cache_reuse_gbps=cache_gbps,
                    latency_per_step_ms=(t_loop / steps) * 1000,
                )
            )

    report.summary = build_summary(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== SD UNet Sampling Benchmark ===")
    for k, v in report.summary.items():
        print(f"{k}: {v}")
    print("Report saved to:", args.output)


if __name__ == "__main__":
    main()
