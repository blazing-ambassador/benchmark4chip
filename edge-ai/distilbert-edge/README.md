# DistilBERT 端侧小 LLM 推理

> 片上存储极限约束测试

## 所属分类

[edge-ai](../README.md)

## 算法说明

极简 PyTorch 负载，模拟 **DistilBERT 端侧小 LLM 推理** 场景并采集与 overview 对齐的架构导向代理指标。

## 硬件考察维度

| 指标 | 含义 | 采集方式 |
|------|------|----------|
| 权重压缩后片上 Flash 加载速度 | 架构代理指标 | sweep / 计时采集 |
| 极小缓存下 Attention 分块计算 | 架构代理指标 | sweep / 计时采集 |
| 内存带宽受限场景算力衰减率 | 架构代理指标 | sweep / 计时采集 |

## 快速运行

```bash
cd edge-ai/distilbert-edge
pip install -r requirements.txt
python src/run.py
```

```bash
python src/run.py --device auto --warmup 5 --iterations 20
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
distilbert-edge/
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
