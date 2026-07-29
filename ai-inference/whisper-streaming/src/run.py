#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from bench_utils import onchip_throughput_gbps, resolve_device, sync_device, timed_call

from metrics import BenchmarkReport, SweepPoint, build_summary
from model import MiniWhisperEncoder


def parse_args():
    p = argparse.ArgumentParser(description="Whisper streaming benchmark")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--window-sizes", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--total-frames", type=int, default=2048)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "results" / "benchmark_report.json")
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    model = MiniWhisperEncoder().to(device).eval()
    report = BenchmarkReport(device=str(device))

    for window in args.window_sizes:
        for batch in args.batch_sizes:
            mel = torch.randn(batch, 80, window, device=device)

            def one_window():
                with torch.no_grad():
                    model(mel)

            for _ in range(args.warmup):
                one_window()
                sync_device(device)

            t_win = timed_call(one_window, args.iterations, device)

            def streaming():
                pos = 0
                with torch.no_grad():
                    while pos < args.total_frames:
                        end = min(pos + window, args.total_frames)
                        chunk = torch.randn(batch, 80, end - pos, device=device)
                        model(chunk)
                        pos = end

            for _ in range(args.warmup):
                streaming()
                sync_device(device)

            t_stream = timed_call(streaming, max(1, args.iterations // 2), device)
            num_windows = max((args.total_frames + window - 1) // window, 1)
            seg_fps = num_windows / t_stream
            ctx_mb = model.context_bytes(batch, args.total_frames) / (1024 * 1024)
            io_bytes = batch * 80 * args.total_frames * 4
            io_gbps = onchip_throughput_gbps(io_bytes, t_stream)

            report.sweep_points.append(
                SweepPoint(
                    batch_size=batch,
                    window_size=window,
                    total_frames=args.total_frames,
                    sliding_window_segment_throughput_fps=seg_fps,
                    context_memory_growth_mb=ctx_mb,
                    temporal_attention_latency_ms=t_win * 1000,
                    streaming_io_throughput_gbps=io_gbps,
                )
            )

    report.summary = build_summary(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Whisper Streaming Benchmark ===")
    for k, v in report.summary.items():
        print(f"{k}: {v}")
    print("Report saved to:", args.output)


if __name__ == "__main__":
    main()
