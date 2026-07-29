# 云端 AI Benchmark

数据中心大规模集群场景，考察多芯片互联、稀疏推理、租户隔离与训推一体能力。

## Benchmark 列表

| 目录 | 算法 | 状态 |
|------|------|------|
| [distributed-tp-inference](./distributed-tp-inference/) | 大模型分布式张量并行推理 | implemented |
| [moe-cloud-batch](./moe-cloud-batch/) | MoE 稀疏专家云端批量推理 | implemented |
| [multi-model-tenant](./multi-model-tenant/) | 多模型混部租户压测 | implemented |
| [clip-distributed](./clip-distributed/) | CLIP 海量图片分布式特征抽取 | implemented |
| [embedding-concurrent](./embedding-concurrent/) | Embedding 高并发向量推理 | implemented |
| [train-infer-fp8-e2e](./train-infer-fp8-e2e/) | 训推一体 FP8 全链路负载 | implemented |

## 统一运行方式

每个 benchmark 目录结构一致：

```
<benchmark>/
├── src/run.py
├── requirements.txt
└── results/benchmark_report.json
```

```bash
cd cloud-ai/<benchmark>
pip install -r requirements.txt
python src/run.py
```

公共工具见 [common/bench_utils.py](../../common/bench_utils.py)（设备选择、计时、吞吐计算）。

## 使用说明

1. 进入对应 benchmark 目录查看 `README.md` 与 `benchmark.yaml`
2. 在目标芯片/设备上运行 `src/run.py` 采集指标
3. 根据 `results/benchmark_report.json` 输出架构对标报告
