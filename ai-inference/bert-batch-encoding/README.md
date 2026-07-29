# BERT 批量向量编码推理

> 小 Transformer 高并发吞吐压力

## 所属分类

[ai-inference](../README.md)

## 算法说明

极简 BERT 编码器（2 层 Transformer + mean pooling），模拟 **短序列、高 batch、高并发** 的向量编码推理场景，用于采集架构导向指标。

## 硬件考察维度

| 指标 | 含义 | 采集方式 |
|------|------|----------|
| 批量并行计算单元利用率 | batch 增大时单样本吞吐是否线性扩展 | `(throughput@B / B) / throughput@1` |
| 短序列算子调度效率 | 短 seq_len 下算力单元是否被充分填满 | `achieved_FLOPs / (peak_FLOPs × latency)` |
| 高并发下算力溢出阈值 | batch / 并发流饱和拐点 | batch 与 concurrency sweep 边际增益 < 5% |
| 片上缓存批量数据吞吐 | 权重+激活访存压力 | `estimated_bytes / latency` (GB/s) |

## 快速运行

```bash
cd ai-inference/bert-batch-encoding
pip install -r requirements.txt
python src/run.py
```

可选参数：

```bash
python src/run.py --device auto --batch-sizes 1 2 4 8 16 32 64 --seq-lens 16 32 64 --concurrency-levels 1 2 4 8 --peak-flops 1e12
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
bert-batch-encoding/
├── README.md
├── benchmark.yaml
├── requirements.txt
├── results/
│   └── benchmark_report.json
└── src/
    ├── model.py      # MiniBertEncoder
    ├── metrics.py    # 四项架构指标
    └── run.py        # 运行入口
```

## 运行状态

- [x] 负载实现
- [x] 指标采集
- [ ] 架构对标报告
