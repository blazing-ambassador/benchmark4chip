# 训推一体 FP8 全链路负载

> 混合精度端到端硬件适配

## 所属分类

[cloud-ai](../README.md)

## 算法说明

极简 PyTorch 负载，模拟 **训推一体 FP8 全链路负载** 场景并采集与 overview 对齐的架构导向代理指标。

## 硬件考察维度

| 指标 | 含义 | 采集方式 |
|------|------|----------|
| FP8 训推统一数据通路 | 架构代理指标 | sweep / 计时采集 |
| 训练量化→推理加载无缝流转 | 架构代理指标 | sweep / 计时采集 |
| 混合精度存储带宽综合利用率 | 架构代理指标 | sweep / 计时采集 |

## 快速运行

```bash
cd cloud-ai/train-infer-fp8-e2e
pip install -r requirements.txt
python src/run.py
```

```bash
python src/run.py --device auto --warmup 5 --iterations 20
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
train-infer-fp8-e2e/
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
