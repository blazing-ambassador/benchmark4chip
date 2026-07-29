# 大模型分布式张量并行推理

> MCM 多芯片互联与规约通信

## 所属分类

[cloud-ai](../README.md)

## 算法说明

极简 PyTorch 负载，模拟 **大模型分布式张量并行推理** 场景并采集与 overview 对齐的架构导向代理指标。

## 硬件考察维度

| 指标 | 含义 | 采集方式 |
|------|------|----------|
| 多 die 高速互联带宽 | 架构代理指标 | sweep / 计时采集 |
| 推理阶段 AllReduce 规约延迟 | 架构代理指标 | sweep / 计时采集 |
| 片间 NOC 拥塞控制 | 架构代理指标 | sweep / 计时采集 |
| 多芯片负载均衡调度 | 架构代理指标 | sweep / 计时采集 |

## 快速运行

```bash
cd cloud-ai/distributed-tp-inference
pip install -r requirements.txt
python src/run.py
```

```bash
python src/run.py --device auto --warmup 5 --iterations 20
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
distributed-tp-inference/
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
