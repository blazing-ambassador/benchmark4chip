# LLaMA-7B 增量解码推理

> KV 缓存复用、自注意力增量计算

## 所属分类

[ai-inference](../README.md)

## 硬件考察维度

| 字段 | 指标 | 采集方式 |
|------|------|----------|
| kv_cache_residency_ratio | 片上高速缓存对 KV Cache 驻留能力 | min(cache, kv_bytes)/kv_bytes |
| seq_length_compute_elasticity | 动态序列长度算力弹性 | per_sample_tps / baseline_tps |
| time_to_first_token_ms | 解码时延/首包响应延迟 | prefill latency |
| batch_scheduling_utilization | 批量推理调度单元 | batch 扩展效率 |
| weight_bandwidth_bottleneck_gbps | 权重带宽瓶颈 | weight_bytes/latency |

## 快速运行

```bash
cd ai-inference/llama-7b-decode
pip install -r requirements.txt
python src/run.py
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
llama-7b-decode/
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
