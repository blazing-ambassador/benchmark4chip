# YOLOv8-Nano 实时检测推理

> 轻量化卷积 + 后处理 NPU 算子

## 所属分类

[edge-ai](../README.md)

## 算法说明

极简 PyTorch 负载，模拟 **YOLOv8-Nano 实时检测推理** 场景并采集与 overview 对齐的架构导向代理指标。

## 硬件考察维度

| 指标 | 含义 | 采集方式 |
|------|------|----------|
| 深度可分离卷积硬件效率 | 架构代理指标 | sweep / 计时采集 |
| NMS/置信度筛选后处理硬加速 | 架构代理指标 | sweep / 计时采集 |
| 低功耗模式算力输出 | 架构代理指标 | sweep / 计时采集 |
| 片内小容量 SRAM 利用率 | 架构代理指标 | sweep / 计时采集 |

## 快速运行

```bash
cd edge-ai/yolov8-nano
pip install -r requirements.txt
python src/run.py
```

```bash
python src/run.py --device auto --warmup 5 --iterations 20
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
yolov8-nano/
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
