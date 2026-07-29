#!/usr/bin/env python3
"""Minimal BERT batch encoding benchmark runner."""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

import torch

from metrics import (
    BenchmarkReport,
    SweepPoint,
    build_summary,
    compute_batch_parallel_utilization,
    compute_onchip_cache_throughput_gbps,
    compute_short_seq_scheduling_efficiency,
    find_concurrency_overflow_threshold,
    find_overflow_threshold,
)
from model import MiniBertEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BERT batch encoding micro-benchmark")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--concurrency-levels", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--peak-flops", type=float, default=0.0, help="Device peak FLOPS/s for efficiency baseline")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "benchmark_report.json",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(requested)


def make_inputs(batch_size: int, seq_len: int, vocab_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
    return input_ids, attention_mask


def timed_forward(
    model: MiniBertEncoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    iterations: int,
) -> float:
    """Return average latency in seconds."""
    with torch.no_grad():
        for _ in range(iterations):
            model(input_ids, attention_mask)
            if input_ids.is_cuda:
                torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            model(input_ids, attention_mask)
            if input_ids.is_cuda:
                torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / iterations


def run_concurrent_streams(
    model: MiniBertEncoder,
    batch_size: int,
    seq_len: int,
    concurrency: int,
    iterations: int,
    device: torch.device,
) -> float:
    """Simulate high-concurrency pressure with parallel inference streams."""

    def worker() -> None:
        input_ids, attention_mask = make_inputs(batch_size, seq_len, 30522, device)
        timed_forward(model, input_ids, attention_mask, max(1, iterations // concurrency))

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker) for _ in range(concurrency)]
        for future in futures:
            future.result()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype_bytes = 4

    model = MiniBertEncoder().to(device)
    model.eval()

    peak_flops = args.peak_flops
    if peak_flops <= 0 and device.type == "cuda":
        # Rough FP32 peak for scheduling efficiency baseline when not provided.
        props = torch.cuda.get_device_properties(device)
        peak_flops = props.multi_processor_count * 1.5e12
    if peak_flops <= 0:
        peak_flops = 100e9  # CPU fallback baseline

    report = BenchmarkReport(device=str(device), dtype="float32")
    primary_seq_len = min(args.seq_lens)

    # Warmup at batch=1 before sweep.
    warmup_ids, warmup_mask = make_inputs(1, primary_seq_len, 30522, device)
    for _ in range(args.warmup):
        with torch.no_grad():
            model(warmup_ids, warmup_mask)

    batch_throughputs = []  # type: List[float]
    batch_size_list = []  # type: List[int]
    baseline_throughput = 0.0

    # Batch sweep at shortest sequence length (short-seq scheduling stress).
    for batch_size in args.batch_sizes:
        try:
            input_ids, attention_mask = make_inputs(batch_size, primary_seq_len, 30522, device)
            for _ in range(args.warmup):
                with torch.no_grad():
                    model(input_ids, attention_mask)

            latency_s = timed_forward(model, input_ids, attention_mask, args.iterations)
            throughput = batch_size / latency_s
            if batch_size == 1:
                baseline_throughput = throughput
                report.baseline_throughput_samples_per_s = baseline_throughput

            flops = model.estimate_flops_per_forward(batch_size, primary_seq_len)
            bytes_moved = model.estimate_bytes_per_forward(batch_size, primary_seq_len, dtype_bytes)

            point = SweepPoint(
                batch_size=batch_size,
                seq_len=primary_seq_len,
                latency_ms=latency_s * 1000,
                throughput_samples_per_s=throughput,
                throughput_tokens_per_s=throughput * primary_seq_len,
                batch_parallel_utilization=compute_batch_parallel_utilization(
                    throughput, batch_size, baseline_throughput
                ),
                short_seq_scheduling_efficiency=compute_short_seq_scheduling_efficiency(
                    latency_s, primary_seq_len, batch_size, flops, peak_flops
                ),
                onchip_cache_batch_throughput_gbps=compute_onchip_cache_throughput_gbps(
                    bytes_moved, latency_s
                ),
            )
            report.sweep_points.append(point)
            batch_throughputs.append(throughput)
            batch_size_list.append(batch_size)
        except RuntimeError as exc:
            print(f"Skip batch_size={batch_size}: {exc}", file=sys.stderr)
            break

    report.compute_overflow_threshold_batch = find_overflow_threshold(
        batch_size_list, batch_throughputs
    )

    # Concurrency sweep at moderate batch to find overflow under parallel load.
    concurrency_throughputs = []  # type: List[float]
    test_batch = min(8, max(batch_size_list) if batch_size_list else 8)
    for concurrency in args.concurrency_levels:
        total_time = run_concurrent_streams(
            model, test_batch, primary_seq_len, concurrency, args.iterations, device
        )
        total_samples = test_batch * concurrency
        concurrency_throughputs.append(total_samples / total_time)

    report.compute_overflow_threshold_concurrency = find_concurrency_overflow_threshold(
        args.concurrency_levels[: len(concurrency_throughputs)],
        concurrency_throughputs,
    )

    # Additional short-seq length points at fixed batch.
    fixed_batch = min(16, max(batch_size_list) if batch_size_list else 16)
    for seq_len in args.seq_lens:
        if seq_len == primary_seq_len:
            continue
        try:
            input_ids, attention_mask = make_inputs(fixed_batch, seq_len, 30522, device)
            latency_s = timed_forward(model, input_ids, attention_mask, args.iterations)
            throughput = fixed_batch / latency_s
            flops = model.estimate_flops_per_forward(fixed_batch, seq_len)
            bytes_moved = model.estimate_bytes_per_forward(fixed_batch, seq_len, dtype_bytes)

            report.sweep_points.append(
                SweepPoint(
                    batch_size=fixed_batch,
                    seq_len=seq_len,
                    latency_ms=latency_s * 1000,
                    throughput_samples_per_s=throughput,
                    throughput_tokens_per_s=throughput * seq_len,
                    batch_parallel_utilization=compute_batch_parallel_utilization(
                        throughput, fixed_batch, baseline_throughput
                    ),
                    short_seq_scheduling_efficiency=compute_short_seq_scheduling_efficiency(
                        latency_s, seq_len, fixed_batch, flops, peak_flops
                    ),
                    onchip_cache_batch_throughput_gbps=compute_onchip_cache_throughput_gbps(
                        bytes_moved, latency_s
                    ),
                )
            )
        except RuntimeError as exc:
            print(f"Skip seq_len={seq_len}: {exc}", file=sys.stderr)

    report.summary = build_summary(report)
    report.summary["concurrency_throughput_samples_per_s"] = {
        str(c): round(t, 2)
        for c, t in zip(args.concurrency_levels[: len(concurrency_throughputs)], concurrency_throughputs)
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== BERT Batch Encoding Benchmark ===")
    print(f"Device: {device}")
    print(f"Baseline throughput (batch=1): {report.baseline_throughput_samples_per_s:.2f} samples/s")
    print(f"Batch parallel utilization (peak): {report.summary.get('peak_batch_parallel_utilization', 0):.2%}")
    print(f"Short-seq scheduling efficiency (peak): {report.summary.get('peak_short_seq_scheduling_efficiency', 0):.2%}")
    print(f"Compute overflow threshold (batch): {report.compute_overflow_threshold_batch}")
    print(f"Compute overflow threshold (concurrency): {report.compute_overflow_threshold_concurrency}")
    print(f"On-chip cache batch throughput (peak): {report.summary.get('peak_onchip_cache_batch_throughput_gbps', 0):.3f} GB/s")
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
