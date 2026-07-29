# Qwen-1.8B INT4 量化离线推理

> 权重压缩与带宽瓶颈

## 所属分类

[edge-ai](../README.md)

## 算法说明

极简 PyTorch 负载，模拟 **Qwen-1.8B INT4 量化离线推理** 场景并采集与 overview 对齐的架构导向代理指标。

## 硬件考察维度

| 指标 | 含义 | 采集方式 |
|------|------|----------|
| INT4 解压硬件单元 | 架构代理指标 | sweep / 计时采集 |
| 4bit 权重访存带宽压力 | 架构代理指标 | sweep / 计时采集 |
| 片上缓存分块加载策略 | 架构代理指标 | sweep / 计时采集 |
| 低比特数值计算稳定性 | 架构代理指标 | sweep / 计时采集 |

## 快速运行

```bash
cd edge-ai/qwen-int4
pip install -r requirements.txt
python src/run.py
```

```bash
python src/run.py --device auto --warmup 5 --iterations 20
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
qwen-int4/
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
