# benchmark-template

复制本目录到 `<category>/<new-benchmark-id>/`，并重命名占位符。

## 文件

| 文件 | 说明 |
|------|------|
| README.md | 算法与指标说明 |
| benchmark.yaml | 元数据 |
| requirements.txt | 依赖 |
| src/model.py | 负载 |
| src/metrics.py | 指标 |
| src/run.py | 入口（`sys.path` 指向仓库根 `common/`） |

运行前在 `results/` 下放置 `.gitkeep`（可选）；报告输出为 `results/benchmark_report.json`。

参考实现：[ai-inference/bert-batch-encoding](../../ai-inference/bert-batch-encoding/)。
