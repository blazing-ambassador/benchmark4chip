# CLIP 海量图片分布式特征抽取

> IO + 算力满负载跑批

## 所属分类

[cloud-ai](../README.md)

## 算法说明

极简 PyTorch 负载，模拟 **CLIP 海量图片分布式特征抽取** 场景并采集与 overview 对齐的架构导向代理指标。

## 硬件考察维度

| 指标 | 含义 | 采集方式 |
|------|------|----------|
| PCIe/主机 DMA 带宽上限 | 架构代理指标 | sweep / 计时采集 |
| 批量图像预处理硬加速 | 架构代理指标 | sweep / 计时采集 |
| 算力持续满载稳定性 | 架构代理指标 | sweep / 计时采集 |
| DRAM 持续读写带宽 | 架构代理指标 | sweep / 计时采集 |

## 快速运行

```bash
cd cloud-ai/clip-distributed
pip install -r requirements.txt
python src/run.py
```

```bash
python src/run.py --device auto --warmup 5 --iterations 20
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
clip-distributed/
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
