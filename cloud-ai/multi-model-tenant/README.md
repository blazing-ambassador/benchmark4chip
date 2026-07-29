# 多模型混部租户压测

> 算力调度与资源隔离开销

## 所属分类

[cloud-ai](../README.md)

## 算法说明

极简 PyTorch 负载，模拟 **多模型混部租户压测** 场景并采集与 overview 对齐的架构导向代理指标。

## 硬件考察维度

| 指标 | 含义 | 采集方式 |
|------|------|----------|
| 硬件资源虚拟化隔离 | 架构代理指标 | sweep / 计时采集 |
| 算力切片抢占调度 | 架构代理指标 | sweep / 计时采集 |
| 缓存分区保护 | 架构代理指标 | sweep / 计时采集 |
| 多任务 QoS 时延保障能力 | 架构代理指标 | sweep / 计时采集 |

## 快速运行

```bash
cd cloud-ai/multi-model-tenant
pip install -r requirements.txt
python src/run.py
```

```bash
python src/run.py --device auto --warmup 5 --iterations 20
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
multi-model-tenant/
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
