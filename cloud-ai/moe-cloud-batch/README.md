# MoE 稀疏专家云端批量推理

> 动态路由负载不均衡访存

## 所属分类

[cloud-ai](../README.md)

## 算法说明

极简 PyTorch 负载，模拟 **MoE 稀疏专家云端批量推理** 场景并采集与 overview 对齐的架构导向代理指标。

## 硬件考察维度

| 指标 | 含义 | 采集方式 |
|------|------|----------|
| 稀疏 Token 动态分发硬件单元 | 架构代理指标 | sweep / 计时采集 |
| 专家核算力动态调度 | 架构代理指标 | sweep / 计时采集 |
| 不规则访存带宽冲突规避 | 架构代理指标 | sweep / 计时采集 |
| 批量任务负载打散能力 | 架构代理指标 | sweep / 计时采集 |

## 快速运行

```bash
cd cloud-ai/moe-cloud-batch
pip install -r requirements.txt
python src/run.py
```

```bash
python src/run.py --device auto --warmup 5 --iterations 20
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
moe-cloud-batch/
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
