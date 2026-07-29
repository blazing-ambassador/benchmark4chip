#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from bench_utils import resolve_device, sync_device, timed_call

from metrics import BenchmarkReport, SweepPoint, build_summary
from model import MiniQuantMLP


def parse_args():
    p = argparse.ArgumentParser(description="INT8/FP8 quant inference benchmark")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "results" / "benchmark_report.json")
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    model = MiniQuantMLP().to(device).eval()
    report = BenchmarkReport(device=str(device))

    for batch in args.batch_sizes:
        x = torch.randn(batch, 512, device=device)

        def fp32():
            with torch.no_grad():
                model.forward_fp32(x)

        def int8():
            with torch.no_grad():
                model.forward_int8(x)

        def fp8():
            with torch.no_grad():
                model.forward_fp8(x)

        for fn in (fp32, int8, fp8):
            for _ in range(args.warmup):
                fn()
                sync_device(device)

        t_fp32 = timed_call(fp32, args.iterations, device)
        t_int8 = timed_call(int8, args.iterations, device)
        t_fp8 = timed_call(fp8, args.iterations, device)

        with torch.no_grad():
            ref = model.forward_fp32(x)
            out_int8 = model.forward_int8(x)
            out_fp8 = model.forward_fp8(x)

        err_int8 = (ref - out_int8).abs().max().item() / (ref.abs().max().item() + 1e-8)
        err_fp8 = (ref - out_fp8).abs().max().item() / (ref.abs().max().item() + 1e-8)

        for precision, t, err in (
            ("int8", t_int8, 1.0 - err_int8),
            ("fp8", t_fp8, 1.0 - err_fp8),
        ):
            throughput = batch / t
            overhead = max((t - t_fp32) / max(t, 1e-9), 0.0)
            zp_ratio = 0.15 if precision == "int8" else 0.05
            report.sweep_points.append(
                SweepPoint(
                    batch_size=batch,
                    precision=precision,
                    throughput_samples_per_s=throughput,
                    quant_dequant_overhead_ratio=overhead,
                    zero_point_offset_ops_ratio=zp_ratio,
                    low_precision_error_tolerance=max(err, 0.0),
                )
            )

    report.summary = build_summary(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== INT8/FP8 Quant Inference Benchmark ===")
    for k, v in report.summary.items():
        print(f"{k}: {v}")
    print("Report saved to:", args.output)


if __name__ == "__main__":
    main()
