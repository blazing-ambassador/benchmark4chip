#!/usr/bin/env python3
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from bench_utils import onchip_throughput_gbps, resolve_device, sync_device, timed_call

from metrics import BenchmarkReport, SweepPoint, build_summary
from model import MiniLlavaPipeline


def parse_args():
    p = argparse.ArgumentParser(description="LLaVA multimodal pipeline benchmark")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "results" / "benchmark_report.json")
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    model = MiniLlavaPipeline().to(device).eval()
    report = BenchmarkReport(device=str(device))

    for batch in args.batch_sizes:
        images = torch.randn(batch, 3, 224, 224, device=device)
        text = torch.randn(batch, 8, 256, device=device)

        for _ in range(args.warmup):
            with torch.no_grad():
                model.forward_e2e(images, text)
            sync_device(device)

        def vision():
            with torch.no_grad():
                model.forward_vision(images)

        def project():
            with torch.no_grad():
                v = model.forward_vision(images)
                model.forward_project(v)

        def language():
            with torch.no_grad():
                v = model.forward_vision(images)
                p = model.forward_project(v)
                fused = torch.cat([p, text], dim=1)
                model.forward_language(fused)

        def e2e():
            with torch.no_grad():
                model.forward_e2e(images, text)

        t_v = timed_call(vision, args.iterations, device)

        with torch.no_grad():
            vision_cache = model.forward_vision(images)

        def project_cached():
            with torch.no_grad():
                model.forward_project(vision_cache)

        t_proj_only = timed_call(project_cached, args.iterations, device)
        t_lang = timed_call(language, args.iterations, device)
        t_e2e = timed_call(e2e, args.iterations, device)

        serial = t_v + t_proj_only + max(t_lang - t_v - t_proj_only, 1e-9)
        pipeline_eff = min(serial / t_e2e, 1.0) if t_e2e > 0 else 0.0
        switch_overhead_ms = t_proj_only * 1000
        feat_bytes = model.feature_bytes(batch)
        transfer_gbps = onchip_throughput_gbps(feat_bytes, max(t_proj_only, 1e-6))

        def dual_stream():
            with torch.no_grad():
                model.forward_e2e(images, text)
                model.forward_e2e(images, text)

        t_dual = timed_call(dual_stream, max(1, args.iterations // 2), device)
        isolation_ratio = max((t_dual / (2 * t_e2e)) - 1.0, 0.0)

        report.sweep_points.append(
            SweepPoint(
                batch_size=batch,
                pipeline_parallel_efficiency=pipeline_eff,
                model_switch_overhead_ms=switch_overhead_ms,
                feature_projection_transfer_gbps=transfer_gbps,
                multitask_isolation_overhead_ratio=isolation_ratio,
                end_to_end_latency_ms=t_e2e * 1000,
            )
        )

    report.summary = build_summary(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== LLaVA Multimodal Benchmark ===")
    for k, v in report.summary.items():
        print(f"{k}: {v}")
    print("Report saved to:", args.output)


if __name__ == "__main__":
    main()
