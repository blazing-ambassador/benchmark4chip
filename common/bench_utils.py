"""Shared helpers for Benchmark4Chip micro-benchmarks."""

import sys
import time
from typing import List

import torch


def repo_common_dir(from_file: str) -> str:
    """Return path to repository ``common/`` from ``.../<category>/<bench>/src/run.py``."""
    from pathlib import Path

    return str(Path(from_file).resolve().parents[3] / "common")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(requested)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed_call(fn, iterations: int, device: torch.device) -> float:
    """Average latency in seconds for fn()."""
    for _ in range(iterations):
        fn()
        sync_device(device)

    start = time.perf_counter()
    for _ in range(iterations):
        fn()
        sync_device(device)
    return (time.perf_counter() - start) / iterations


def onchip_throughput_gbps(bytes_moved: int, latency_s: float) -> float:
    if latency_s <= 0:
        return 0.0
    return bytes_moved / latency_s / 1e9


def find_saturation_knee(values: List[int], throughputs: List[float], growth_ratio: float = 0.05) -> int:
    if not values:
        return 0
    best_v = values[0]
    best_t = throughputs[0]
    for v, t in zip(values, throughputs):
        if t > best_t * (1.0 + growth_ratio):
            best_t = t
            best_v = v
    return best_v


def peak_flops_default(device: torch.device, user_peak: float) -> float:
    if user_peak > 0:
        return user_peak
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return props.multi_processor_count * 1.5e12
    return 100e9
