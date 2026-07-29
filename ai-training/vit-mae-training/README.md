# ViT-MAE 视觉自监督训练

> 纯视觉 Transformer

## 所属分类

[ai-training](../README.md)

## 算法说明

极简 PyTorch 负载，模拟 **ViT-MAE 视觉自监督训练** 场景并采集与 overview 对齐的架构导向代理指标。

## 硬件考察维度

| 指标 | 含义 | 采集方式 |
|------|------|----------|
| 长序列注意力矩阵乘算力 | 架构代理指标 | sweep / 计时采集 |
| Patch 张量重排开销 | 架构代理指标 | sweep / 计时采集 |
| 大维度张量片内搬运 | 架构代理指标 | sweep / 计时采集 |
| 归一化层硬件加速 | 架构代理指标 | sweep / 计时采集 |
| 高带宽 DRAM 吞吐 | 架构代理指标 | sweep / 计时采集 |

## 快速运行

```bash
cd ai-training/vit-mae-training
pip install -r requirements.txt
python src/run.py
```

```bash
python src/run.py --device auto --warmup 5 --iterations 20
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
vit-mae-training/
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
