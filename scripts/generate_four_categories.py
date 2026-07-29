#!/usr/bin/env python3
"""Generate runnable implementations for four benchmark categories."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from generate_benchmarks import CATEGORIES, yaml_list  # noqa: E402
from category_archetypes import get_benchmark_sources  # noqa: E402

TARGET_CATEGORIES = ("ai-training", "edge-ai", "cloud-ai", "scientific-computing")

REQUIREMENTS = "torch>=1.9.0\n"


def write_implemented_benchmark(category: str, bench: dict) -> None:
    bench_id = bench["id"]
    bench_dir = ROOT / category / bench_id
    bench_dir.mkdir(parents=True, exist_ok=True)

    model_py, metrics_py, run_py = get_benchmark_sources(bench_id, bench)

    src = bench_dir / "src"
    src.mkdir(exist_ok=True)
    (src / "model.py").write_text(model_py, encoding="utf-8")
    (src / "metrics.py").write_text(metrics_py, encoding="utf-8")
    (src / "run.py").write_text(run_py, encoding="utf-8")
    (bench_dir / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (bench_dir / "results").mkdir(exist_ok=True)

    metric_table = "\n".join(
        "| {0} | 架构代理指标 | sweep / 计时采集 |".format(m) for m in bench["metrics"]
    )

    readme = """# {name}

> {summary}

## 所属分类

[{category}](../README.md)

## 算法说明

极简 PyTorch 负载，模拟 **{name}** 场景并采集与 overview 对齐的架构导向代理指标。

## 硬件考察维度

| 指标 | 含义 | 采集方式 |
|------|------|----------|
{metric_table}

## 快速运行

```bash
cd {category}/{bench_id}
pip install -r requirements.txt
python src/run.py
```

```bash
python src/run.py --device auto --warmup 5 --iterations 20
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
{bench_id}/
├── README.md
├── benchmark.yaml
├── requirements.txt
├── results/
│   └── benchmark_report.json
└── src/
    ├── model.py
    ├── metrics.py
    └── run.py
```

## 运行状态

- [x] 负载实现
- [x] 指标采集
- [ ] 架构对标报告
""".format(
        name=bench["name"],
        summary=bench["summary"],
        category=category,
        bench_id=bench_id,
        metric_table=metric_table,
    )
    (bench_dir / "README.md").write_text(readme, encoding="utf-8")

    yaml_content = """id: {id}
name: {name}
category: {category}
summary: {summary}
status: implemented
entrypoint: src/run.py
requirements: requirements.txt
output: results/benchmark_report.json
metrics:
{metrics}
tags:
  - chip-benchmark
  - architecture-validation
""".format(
        id=bench_id,
        name=bench["name"],
        category=category,
        summary=bench["summary"],
        metrics=yaml_list(bench["metrics"], 2),
    )
    (bench_dir / "benchmark.yaml").write_text(yaml_content, encoding="utf-8")


def write_category_readme(category: str, info: dict) -> None:
    table_rows = "\n".join(
        "| [{id}](./{id}/) | {name} | implemented |".format(id=b["id"], name=b["name"])
        for b in info["benchmarks"]
    )
    readme = """# {title}

{description}

## Benchmark 列表

| 目录 | 算法 | 状态 |
|------|------|------|
{table_rows}

## 统一运行方式

每个 benchmark 目录结构一致：

```
<benchmark>/
├── src/run.py
├── requirements.txt
└── results/benchmark_report.json
```

```bash
cd {category}/<benchmark>
pip install -r requirements.txt
python src/run.py
```

公共工具见 [common/bench_utils.py](../../common/bench_utils.py)（设备选择、计时、吞吐计算）。

## 使用说明

1. 进入对应 benchmark 目录查看 `README.md` 与 `benchmark.yaml`
2. 在目标芯片/设备上运行 `src/run.py` 采集指标
3. 根据 `results/benchmark_report.json` 输出架构对标报告
""".format(
        title=info["title"],
        description=info["description"],
        table_rows=table_rows,
        category=category,
    )
    (ROOT / category / "README.md").write_text(readme, encoding="utf-8")


def update_root_readme() -> None:
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    old = """## Benchmark 组织方式

每个 benchmark 目录包含：

```
<benchmark-name>/
├── README.md        # 算法说明、硬件考察维度、运行指引
└── benchmark.yaml   # 结构化元数据（指标、标签、状态）
```

新增 benchmark 可复制 [common/benchmark-template](./common/benchmark-template/) 模板。"""
    new = """## Benchmark 组织方式

每个 benchmark 目录包含：

```
<benchmark-name>/
├── README.md           # 算法说明、硬件考察维度、运行指引
├── benchmark.yaml      # 结构化元数据（指标、标签、状态）
├── requirements.txt    # Python 依赖（torch>=1.9.0）
├── results/            # benchmark_report.json 输出
└── src/                # model.py、metrics.py、run.py 可运行负载
```

`ai-training`、`ai-inference`、`edge-ai`、`cloud-ai`、`scientific-computing` 五类目录下的 benchmark 均提供可运行的 `src/` 实现。

新增 benchmark 可复制 [common/benchmark-template](./common/benchmark-template/) 模板。"""
    if old in text:
        text = text.replace(old, new)
    else:
        if "均提供可运行的 `src/` 实现" not in text:
            text = text.rstrip() + "\n\n" + new.split("\n\n", 1)[-1] + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    count = 0
    for category in TARGET_CATEGORIES:
        info = CATEGORIES[category]
        for bench in info["benchmarks"]:
            write_implemented_benchmark(category, bench)
            count += 1
        write_category_readme(category, info)
    print("Generated {0} runnable benchmarks in {1} categories.".format(count, len(TARGET_CATEGORIES)))


if __name__ == "__main__":
    main()
