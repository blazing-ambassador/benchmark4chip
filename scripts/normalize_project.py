#!/usr/bin/env python3
"""Normalize repository layout: common paths, category README links, optional cleanup."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OLD_SNIPPETS = (
    'sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))',
    "sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'common'))",
)
NEW_SNIPPET = (
    'sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))'
)

CATEGORY_README_OLD = "公共工具见 [common/bench_utils.py](./common/bench_utils.py)"
CATEGORY_README_NEW = "公共工具见 [common/bench_utils.py](../../common/bench_utils.py)"


def patch_run_py(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    orig = text
    for old in OLD_SNIPPETS:
        text = text.replace(old, NEW_SNIPPET)
    if text != orig:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_category_readmes() -> int:
    n = 0
    for category in ("ai-training", "ai-inference", "edge-ai", "cloud-ai", "scientific-computing"):
        readme = ROOT / category / "README.md"
        if not readme.exists():
            continue
        text = readme.read_text(encoding="utf-8")
        if CATEGORY_README_OLD in text:
            readme.write_text(text.replace(CATEGORY_README_OLD, CATEGORY_README_NEW), encoding="utf-8")
            n += 1
    return n


def remove_redundant_category_common() -> int:
    removed = 0
    for category in ("ai-training", "ai-inference", "edge-ai", "cloud-ai", "scientific-computing"):
        p = ROOT / category / "common" / "bench_utils.py"
        if p.exists():
            p.unlink()
            removed += 1
            common_dir = p.parent
            if common_dir.exists() and not any(common_dir.iterdir()):
                common_dir.rmdir()
    return removed


def main() -> None:
    runs = list(ROOT.glob("*/*/src/run.py"))
    patched = sum(1 for p in runs if patch_run_py(p))
    readmes = patch_category_readmes()
    removed = remove_redundant_category_common()
    print("Patched run.py:", patched)
    print("Patched category README:", readmes)
    print("Removed redundant bench_utils copies:", removed)


if __name__ == "__main__":
    main()
