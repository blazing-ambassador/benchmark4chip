#!/usr/bin/env python3
"""LLaMA-style incremental decode micro-benchmark."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from bench_utils import peak_flops_default, resolve_device, sync_device, timed_call

from metrics import (
    BenchmarkReport,
    SweepPoint,
    batch_scheduling_utilization,
    build_summary,
    kv_cache_residency_ratio,
    seq_length_compute_elasticity,
    weight_bandwidth_bottleneck_gbps,
)
from model import MiniLlamaDecoder


def parse_args():
    parser = argparse.ArgumentParser(description="LLaMA incremental decode benchmark")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--context-lens", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--onchip-cache-kb", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--peak-flops", type=float, default=0.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "benchmark_report.json",
    )
    return parser.parse_args()


def run_decode_episode(model, device, batch_size, context_len, decode_steps):
    model.eval()
    cache_bytes = 512 * 1024  # filled by caller via report; residency uses args in main

    def episode():
        with torch.no_grad():
            prefill = torch.randint(0, 32000, (batch_size, context_len), device=device)
            logits, past = model(prefill)
            _ = logits
            token = torch.randint(0, 32000, (batch_size, 1), device=device)
            for _ in range(decode_steps - 1):
                logits, past = model(token, past)
                token = logits.argmax(dim=-1)

    return episode


def main():
    args = parse_args()
    device = resolve_device(args.device)
    _ = peak_flops_default(device, args.peak_flops)

    model = MiniLlamaDecoder().to(device)
    cache_bytes = args.onchip_cache_kb * 1024
    report = BenchmarkReport(device=str(device), simulated_onchip_cache_kb=args.onchip_cache_kb)

    baseline_per_sample_tps = 0.0

    for ctx in args.context_lens:
        for batch in args.batch_sizes:
            episode_fn = run_decode_episode(model, device, batch, ctx, args.decode_steps)

            for _ in range(args.warmup):
                episode_fn()
                sync_device(device)

            def one_ep():
                episode_fn()

            latency = timed_call(one_ep, args.iterations, device)
            total_tokens = batch * (ctx + args.decode_steps)
            tps = total_tokens / latency
            per_sample_tps = tps / batch

            if ctx == args.context_lens[0] and batch == 1:
                baseline_per_sample_tps = per_sample_tps

            # TTFT proxy: prefill-only timing
            def prefill_only():
                with torch.no_grad():
                    prefill = torch.randint(0, 32000, (batch, ctx), device=device)
                    model(prefill)

            ttft_s = timed_call(prefill_only, max(1, args.iterations // 2), device)
            per_token_s = max(latency - ttft_s, 1e-9) / max(args.decode_steps, 1)

            kv_bytes = model.kv_cache_bytes(batch, ctx + args.decode_steps)
            weight_bytes = model.weight_bytes()

            report.sweep_points.append(
                SweepPoint(
                    batch_size=batch,
                    context_len=ctx,
                    decode_steps=args.decode_steps,
                    time_to_first_token_ms=ttft_s * 1000,
                    per_token_decode_latency_ms=per_token_s * 1000,
                    kv_cache_residency_ratio=kv_cache_residency_ratio(kv_bytes, cache_bytes),
                    seq_length_compute_elasticity=seq_length_compute_elasticity(baseline_per_sample_tps, per_sample_tps),
                    batch_scheduling_utilization=batch_scheduling_utilization(per_sample_tps, baseline_per_sample_tps),
                    weight_bandwidth_bottleneck_gbps=weight_bandwidth_bottleneck_gbps(weight_bytes, latency),
                )
            )

    report.summary = build_summary(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== LLaMA Incremental Decode Benchmark ===")
    print("Device:", device)
    for k, v in report.summary.items():
        print(f"{k}: {v}")
    print("Report saved to:", args.output)


if __name__ == "__main__":
    main()
