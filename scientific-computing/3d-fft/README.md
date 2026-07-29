# 3D-FFT 大规模傅里叶变换

> 内存带宽极限压榨

## 所属分类

[scientific-computing](../README.md)

## 算法说明

极简 PyTorch 负载，模拟 **3D-FFT 大规模傅里叶变换** 场景并采集与 overview 对齐的架构导向代理指标。

## 硬件考察维度

| 指标 | 含义 | 采集方式 |
|------|------|----------|
| 多维张量转置硬件开销 | 架构代理指标 | sweep / 计时采集 |
| 跨步访存带宽利用率 | 架构代理指标 | sweep / 计时采集 |
| FFT 蝶形运算单元加速 | 架构代理指标 | sweep / 计时采集 |
| 大尺寸数组缓存分块能力 | 架构代理指标 | sweep / 计时采集 |

## 快速运行

```bash
cd scientific-computing/3d-fft
pip install -r requirements.txt
python src/run.py
```

```bash
python src/run.py --device auto --warmup 5 --iterations 20
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
3d-fft/
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
