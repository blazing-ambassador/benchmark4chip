# Whisper 流式长音频推理

> 时序滑动窗口超长序列计算

## 所属分类

[ai-inference](../README.md)

## 硬件考察维度

| 字段 | 指标 | 采集方式 |
|------|------|----------|
| sliding_window_segment_throughput_fps | 滑动窗口张量分段处理能力 | windows/sec |
| context_memory_growth_mb | 超长上下文显存占用控制 | context_bytes |
| temporal_attention_latency_ms | 时序类 Attention 硬件延迟 | 单窗口 attention 延迟 |
| streaming_io_throughput_gbps | 流式数据 IO 吞吐 | mel_bytes/latency |

## 快速运行

```bash
cd ai-inference/whisper-streaming
pip install -r requirements.txt
python src/run.py
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
whisper-streaming/
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
