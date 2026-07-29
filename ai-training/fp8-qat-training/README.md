# FP8 QAT 量化感知训练

> 混合精度硬件适配

## 所属分类

[ai-training](../README.md)

## 算法说明

极简 PyTorch 负载，模拟 **FP8 QAT 量化感知训练** 场景并采集与 overview 对齐的架构导向代理指标。

## 硬件考察维度

| 指标 | 含义 | 采集方式 |
|------|------|----------|
| FP8 训练原生算力 | 架构代理指标 | sweep / 计时采集 |
| 量化/伪量化算子硬加速 | 架构代理指标 | sweep / 计时采集 |
| 浮点与低精度数据流切换开销 | 架构代理指标 | sweep / 计时采集 |
| 量化误差硬件数值保真度 | 架构代理指标 | sweep / 计时采集 |
| 混合精度存储带宽利用率 | 架构代理指标 | sweep / 计时采集 |

## 快速运行

```bash
cd ai-training/fp8-qat-training
pip install -r requirements.txt
python src/run.py
```

```bash
python src/run.py --device auto --warmup 5 --iterations 20
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
fp8-qat-training/
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
