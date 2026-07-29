# Stable Diffusion UNet 完整训练

> 卷积 + 循环迭代细碎算子

## 所属分类

[ai-training](../README.md)

## 算法说明

极简 PyTorch 负载，模拟 **Stable Diffusion UNet 完整训练** 场景并采集与 overview 对齐的架构导向代理指标。

## 硬件考察维度

| 指标 | 含义 | 采集方式 |
|------|------|----------|
| 小尺寸卷积算子执行效率 | 架构代理指标 | sweep / 计时采集 |
| 算子调度流水线开销 | 架构代理指标 | sweep / 计时采集 |
| 残差分支数据流复用 | 架构代理指标 | sweep / 计时采集 |
| 大量小 Tensor 频繁访存 | 架构代理指标 | sweep / 计时采集 |
| 低精度数值稳定性 | 架构代理指标 | sweep / 计时采集 |

## 快速运行

```bash
cd ai-training/sd-unet-training
pip install -r requirements.txt
python src/run.py
```

```bash
python src/run.py --device auto --warmup 5 --iterations 20
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
sd-unet-training/
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
