# Llava 多模态图文联合推理

> 视觉编码器 + LLM 异构流水线

## 所属分类

[ai-inference](../README.md)

## 硬件考察维度

| 字段 | 指标 | 采集方式 |
|------|------|----------|
| pipeline_parallel_efficiency | 异构算子流水线并行执行能力 | serial_stage_time/e2e_time |
| model_switch_overhead_ms | 多模型数据流无缝切换 | vision→project 切换耗时 |
| feature_projection_transfer_gbps | 特征投影张量传输开销 | feature_bytes/latency |
| multitask_isolation_overhead_ratio | 多任务资源隔离机制 | dual_stream/e2e - 1 |

## 快速运行

```bash
cd ai-inference/llava-multimodal
pip install -r requirements.txt
python src/run.py
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
llava-multimodal/
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
