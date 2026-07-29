# INT8/FP8 量化推理

> 低精度算子与数值稳定性验证

## 所属分类

[ai-inference](../README.md)

## 硬件考察维度

| 字段 | 指标 | 采集方式 |
|------|------|----------|
| throughput_samples_per_s | INT8/FP8 推理算力峰值 | batch/latency |
| quant_dequant_overhead_ratio | 量化反量化硬件开销 | (quant_latency-fp32)/quant |
| zero_point_offset_ops_ratio | 零值点偏移运算单元 | INT8 zero-point 路径占比 proxy |
| low_precision_error_tolerance | 低精度下精度误差容错能力 | 1 - max_relative_error |

## 快速运行

```bash
cd ai-inference/int8-fp8-quant-inference
pip install -r requirements.txt
python src/run.py
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
int8-fp8-quant-inference/
├── README.md
├── benchmark.yaml
├── requirements.txt
├── results/
└── src/
    ├── model.py
    ├── metrics.py
    └── run.py
```

## 运行状态

- [x] 负载实现
- [x] 指标采集
- [ ] 架构对标报告
