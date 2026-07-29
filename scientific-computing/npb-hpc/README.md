# NPB 并行 HPC 基准

> MPI 通信与浮点算力匹配度

## 所属分类

[scientific-computing](../README.md)

## 算法说明

极简 PyTorch 负载，模拟 **NPB 并行 HPC 基准** 场景并采集与 overview 对齐的架构导向代理指标。

## 硬件考察维度

| 指标 | 含义 | 采集方式 |
|------|------|----------|
| 多进程 MPI 通信接口带宽 | 架构代理指标 | sweep / 计时采集 |
| 计算/通信重叠执行能力 | 架构代理指标 | sweep / 计时采集 |
| 浮点运算单元持续效率 | 架构代理指标 | sweep / 计时采集 |
| 并行同步开销 | 架构代理指标 | sweep / 计时采集 |

## 快速运行

```bash
cd scientific-computing/npb-hpc
pip install -r requirements.txt
python src/run.py
```

```bash
python src/run.py --device auto --warmup 5 --iterations 20
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
npb-hpc/
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
