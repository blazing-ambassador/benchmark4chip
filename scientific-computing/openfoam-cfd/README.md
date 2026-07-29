# OpenFOAM CFD 流体求解

> 双精度稠密线性代数算力

## 所属分类

[scientific-computing](../README.md)

## 算法说明

极简 PyTorch 负载，模拟 **OpenFOAM CFD 流体求解** 场景并采集与 overview 对齐的架构导向代理指标。

## 硬件考察维度

| 指标 | 含义 | 采集方式 |
|------|------|----------|
| FP64 双精度峰值算力 | 架构代理指标 | sweep / 计时采集 |
| 稠密矩阵乘阵列效率 | 架构代理指标 | sweep / 计时采集 |
| 迭代求解器访存局部性 | 架构代理指标 | sweep / 计时采集 |
| 大规模方程组存储带宽 | 架构代理指标 | sweep / 计时采集 |

## 快速运行

```bash
cd scientific-computing/openfoam-cfd
pip install -r requirements.txt
python src/run.py
```

```bash
python src/run.py --device auto --warmup 5 --iterations 20
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
openfoam-cfd/
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
