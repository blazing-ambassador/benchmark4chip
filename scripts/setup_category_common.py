#!/usr/bin/env python3
"""Copy root common/bench_utils.py to category/common (legacy shim). Prefer repo root common/."""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "common" / "bench_utils.py"

if not SRC.exists():
    raise SystemExit("Missing common/bench_utils.py at repository root")

for cat in ("ai-training", "ai-inference", "edge-ai", "cloud-ai", "scientific-computing"):
    dst_dir = ROOT / cat / "common"
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC, dst_dir / "bench_utils.py")

print("Synced bench_utils.py from root common/ to category common/ (optional legacy paths)")
