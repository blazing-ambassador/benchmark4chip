# MobileNetV3 分类推理

> 深度可分离卷积低带宽负载

## 所属分类

[edge-ai](../README.md)

## 算法说明

极简 PyTorch 负载，模拟 **MobileNetV3 分类推理** 场景并采集与 overview 对齐的架构导向代理指标。

## 硬件考察维度

| 指标 | 含义 | 采集方式 |
|------|------|----------|
| 卷积数据流寄存器复用 | 架构代理指标 | sweep / 计时采集 |
| 访存带宽节约架构 | 架构代理指标 | sweep / 计时采集 |
| 极低功耗下有效算力 | 架构代理指标 | sweep / 计时采集 |
| MCU 级窄内存接口适配能力 | 架构代理指标 | sweep / 计时采集 |

## 快速运行

```bash
cd edge-ai/mobilenetv3
pip install -r requirements.txt
python src/run.py
```

```bash
python src/run.py --device auto --warmup 5 --iterations 20
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
mobilenetv3/
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
