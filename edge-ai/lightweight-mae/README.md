# 轻量化 MAE 视觉提取推理

> 端侧视觉特征低功耗负载

## 所属分类

[edge-ai](../README.md)

## 算法说明

极简 PyTorch 负载，模拟 **轻量化 MAE 视觉提取推理** 场景并采集与 overview 对齐的架构导向代理指标。

## 硬件考察维度

| 指标 | 含义 | 采集方式 |
|------|------|----------|
| 低帧率算力节流控制 | 架构代理指标 | sweep / 计时采集 |
| 图像 Patch 预处理硬件加速 | 架构代理指标 | sweep / 计时采集 |
| 休眠/运行功耗切换表现 | 架构代理指标 | sweep / 计时采集 |

## 快速运行

```bash
cd edge-ai/lightweight-mae
pip install -r requirements.txt
python src/run.py
```

```bash
python src/run.py --device auto --warmup 5 --iterations 20
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
lightweight-mae/
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
