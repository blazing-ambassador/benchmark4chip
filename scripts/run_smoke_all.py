#!/usr/bin/env python3
"""Run a quick smoke test on one benchmark per category."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SMOKES = [
    ("ai-training", "llama-7b-pretrain"),
    ("ai-inference", "bert-batch-encoding"),
    ("edge-ai", "yolov8-nano"),
    ("cloud-ai", "distributed-tp-inference"),
    ("scientific-computing", "openfoam-cfd"),
]

ARGS = ["--iterations", "2", "--warmup", "1"]


def main() -> int:
    failed = []
    for category, bench_id in SMOKES:
        wd = ROOT / category / bench_id / "src"
        cmd = [sys.executable, "run.py"] + ARGS
        print("=== {0}/{1} ===".format(category, bench_id))
        rc = subprocess.call(cmd, cwd=str(wd))
        if rc != 0:
            failed.append("{0}/{1}".format(category, bench_id))
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1
    print("All smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
