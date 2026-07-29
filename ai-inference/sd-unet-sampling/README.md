# SD UNet 迭代采样推理

> 多步前向细碎卷积算子负载

## 所属分类

[ai-inference](../README.md)

## 硬件考察维度

| 字段 | 指标 | 采集方式 |
|------|------|----------|
| loop_launch_overhead_ratio | 循环多次推理算子启动开销 | (loop - steps*step)/loop |
| conv_array_utilization | 卷积脉动阵列利用率 | achieved_macs/peak_flops |
| intermediate_tensor_cache_reuse_gbps | 中间张量片上缓存复用 | intermediate_bytes/latency |
| latency_per_step_ms | 功耗与时延折中表现 | 每步平均时延 proxy |

## 快速运行

```bash
cd ai-inference/sd-unet-sampling
pip install -r requirements.txt
python src/run.py
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
sd-unet-sampling/
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
