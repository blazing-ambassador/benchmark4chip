# Embedding 高并发向量推理

> 网络吞吐与批量调度压力

## 所属分类

[cloud-ai](../README.md)

## 算法说明

极简 PyTorch 负载，模拟 **Embedding 高并发向量推理** 场景并采集与 overview 对齐的架构导向代理指标。

## 硬件考察维度

| 指标 | 含义 | 采集方式 |
|------|------|----------|
| 超大 Batch 并行算力 | 架构代理指标 | sweep / 计时采集 |
| 向量归一化硬件加速 | 架构代理指标 | sweep / 计时采集 |
| 网卡直连张量传输开销 | 架构代理指标 | sweep / 计时采集 |
| 并发请求队列硬件调度器 | 架构代理指标 | sweep / 计时采集 |

## 快速运行

```bash
cd cloud-ai/embedding-concurrent
pip install -r requirements.txt
python src/run.py
```

```bash
python src/run.py --device auto --warmup 5 --iterations 20
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
embedding-concurrent/
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
