# Megatron 超大模型张量并行模拟

> 多芯片互联

## 所属分类

[ai-training](../README.md)

## 算法说明

极简 PyTorch 负载，模拟 **Megatron 超大模型张量并行模拟** 场景并采集与 overview 对齐的架构导向代理指标。

## 硬件考察维度

| 指标 | 含义 | 采集方式 |
|------|------|----------|
| MCM 多芯片封装高速接口带宽 | 架构代理指标 | sweep / 计时采集 |
| AllReduce/AllGather 规约延迟 | 架构代理指标 | sweep / 计时采集 |
| 多 die NOC 拓扑效率 | 架构代理指标 | sweep / 计时采集 |
| 参数分片流水并行能力 | 架构代理指标 | sweep / 计时采集 |
| 跨芯片同步开销 | 架构代理指标 | sweep / 计时采集 |

## 快速运行

```bash
cd ai-training/megatron-tp-simulation
pip install -r requirements.txt
python src/run.py
```

```bash
python src/run.py --device auto --warmup 5 --iterations 20
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
megatron-tp-simulation/
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
