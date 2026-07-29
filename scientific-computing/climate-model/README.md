# 大气环流气候模式迭代计算

> 大数组持续访存压力

## 所属分类

[scientific-computing](../README.md)

## 算法说明

极简 PyTorch 负载，模拟 **大气环流气候模式迭代计算** 场景并采集与 overview 对齐的架构导向代理指标。

## 硬件考察维度

| 指标 | 含义 | 采集方式 |
|------|------|----------|
| 超大型数组流式读写带宽 | 架构代理指标 | sweep / 计时采集 |
| 长时间步迭代算力稳定性 | 架构代理指标 | sweep / 计时采集 |
| 循环变量寄存器复用 | 架构代理指标 | sweep / 计时采集 |
| 大容量 DRAM 吞吐上限 | 架构代理指标 | sweep / 计时采集 |

## 快速运行

```bash
cd scientific-computing/climate-model
pip install -r requirements.txt
python src/run.py
```

```bash
python src/run.py --device auto --warmup 5 --iterations 20
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
climate-model/
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
