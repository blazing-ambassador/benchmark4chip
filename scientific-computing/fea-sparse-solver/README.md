# FEA 有限元稀疏矩阵求解

> SpMV 稀疏张量运算

## 所属分类

[scientific-computing](../README.md)

## 算法说明

极简 PyTorch 负载，模拟 **FEA 有限元稀疏矩阵求解** 场景并采集与 overview 对齐的架构导向代理指标。

## 硬件考察维度

| 指标 | 含义 | 采集方式 |
|------|------|----------|
| SpMV 稀疏矩阵乘法专用单元 | 架构代理指标 | sweep / 计时采集 |
| 不规则稀疏存储遍历效率 | 架构代理指标 | sweep / 计时采集 |
| 非零元素索引硬件寻址 | 架构代理指标 | sweep / 计时采集 |
| 带宽受限下算力衰减 | 架构代理指标 | sweep / 计时采集 |

## 快速运行

```bash
cd scientific-computing/fea-sparse-solver
pip install -r requirements.txt
python src/run.py
```

```bash
python src/run.py --device auto --warmup 5 --iterations 20
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
fea-sparse-solver/
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
