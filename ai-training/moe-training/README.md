# MoE 混合专家模型训练

> 稀疏动态算力

## 所属分类

[ai-training](../README.md)

## 算法说明

极简 PyTorch 负载，模拟 **MoE 混合专家模型训练** 场景并采集与 overview 对齐的架构导向代理指标。

## 硬件考察维度

| 指标 | 含义 | 采集方式 |
|------|------|----------|
| 动态不规则张量寻址 | 架构代理指标 | sweep / 计时采集 |
| 路由算子硬件加速 | 架构代理指标 | sweep / 计时采集 |
| 动态显存分配效率 | 架构代理指标 | sweep / 计时采集 |
| 专家负载均衡调度能力 | 架构代理指标 | sweep / 计时采集 |
| 稀疏矩阵计算单元效能 | 架构代理指标 | sweep / 计时采集 |
| 跨专家数据搬运带宽 | 架构代理指标 | sweep / 计时采集 |

## 快速运行

```bash
cd ai-training/moe-training
pip install -r requirements.txt
python src/run.py
```

```bash
python src/run.py --device auto --warmup 5 --iterations 20
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
moe-training/
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
