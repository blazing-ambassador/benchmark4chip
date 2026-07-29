# LLaMA-7B 稠密预训练

> 通用 LLM 稠密算力 + 并行通信

## 所属分类

[ai-training](../README.md)

## 算法说明

极简 PyTorch 负载，模拟 **LLaMA-7B 稠密预训练** 场景并采集与 overview 对齐的架构导向代理指标。

## 硬件考察维度

| 指标 | 含义 | 采集方式 |
|------|------|----------|
| BF16/FP16 张量算力利用率 | 架构代理指标 | sweep / 计时采集 |
| 片上 SRAM 缓存命中率 | 架构代理指标 | sweep / 计时采集 |
| 激活重计算显存开销 | 架构代理指标 | sweep / 计时采集 |
| 多卡 TP/PP 集体通信带宽 | 架构代理指标 | sweep / 计时采集 |
| NOC 片上网络吞吐 | 架构代理指标 | sweep / 计时采集 |
| 显存带宽饱和能力 | 架构代理指标 | sweep / 计时采集 |

## 快速运行

```bash
cd ai-training/llama-7b-pretrain
pip install -r requirements.txt
python src/run.py
```

```bash
python src/run.py --device auto --warmup 5 --iterations 20
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
llama-7b-pretrain/
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
