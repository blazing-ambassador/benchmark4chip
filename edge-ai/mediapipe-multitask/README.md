# MediaPipe 多任务串行推理

> 多模型调度开销

## 所属分类

[edge-ai](../README.md)

## 算法说明

极简 PyTorch 负载，模拟 **MediaPipe 多任务串行推理** 场景并采集与 overview 对齐的架构导向代理指标。

## 硬件考察维度

| 指标 | 含义 | 采集方式 |
|------|------|----------|
| 模型快速切换上下文保存/恢复开销 | 架构代理指标 | sweep / 计时采集 |
| 流水线任务硬件调度器 | 架构代理指标 | sweep / 计时采集 |
| 多算子串行流水吞吐 | 架构代理指标 | sweep / 计时采集 |
| 功耗动态调节能力 | 架构代理指标 | sweep / 计时采集 |

## 快速运行

```bash
cd edge-ai/mediapipe-multitask
pip install -r requirements.txt
python src/run.py
```

```bash
python src/run.py --device auto --warmup 5 --iterations 20
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
mediapipe-multitask/
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
