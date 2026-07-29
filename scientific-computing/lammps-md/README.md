# LAMMPS 分子动力学仿真

> 不规则随机内存访问

## 所属分类

[scientific-computing](../README.md)

## 算法说明

极简 PyTorch 负载，模拟 **LAMMPS 分子动力学仿真** 场景并采集与 overview 对齐的架构导向代理指标。

## 硬件考察维度

| 指标 | 含义 | 采集方式 |
|------|------|----------|
| 随机散列访存架构效率 | 架构代理指标 | sweep / 计时采集 |
| 粒子力场计算并行度 | 架构代理指标 | sweep / 计时采集 |
| 非连续地址访问缓存命中率 | 架构代理指标 | sweep / 计时采集 |
| 大数组寻址延迟 | 架构代理指标 | sweep / 计时采集 |

## 快速运行

```bash
cd scientific-computing/lammps-md
pip install -r requirements.txt
python src/run.py
```

```bash
python src/run.py --device auto --warmup 5 --iterations 20
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
lammps-md/
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
