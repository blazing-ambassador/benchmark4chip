#!/usr/bin/env python3
"""Update README and benchmark.yaml for ai-inference benchmarks."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BENCHES = {
    "llama-7b-decode": {
        "title": "LLaMA-7B 增量解码推理",
        "summary": "KV 缓存复用、自注意力增量计算",
        "metrics_table": [
            ("kv_cache_residency_ratio", "片上高速缓存对 KV Cache 驻留能力", "min(cache, kv_bytes)/kv_bytes"),
            ("seq_length_compute_elasticity", "动态序列长度算力弹性", "per_sample_tps / baseline_tps"),
            ("time_to_first_token_ms", "解码时延/首包响应延迟", "prefill latency"),
            ("batch_scheduling_utilization", "批量推理调度单元", "batch 扩展效率"),
            ("weight_bandwidth_bottleneck_gbps", "权重带宽瓶颈", "weight_bytes/latency"),
        ],
    },
    "llava-multimodal": {
        "title": "Llava 多模态图文联合推理",
        "summary": "视觉编码器 + LLM 异构流水线",
        "metrics_table": [
            ("pipeline_parallel_efficiency", "异构算子流水线并行执行能力", "serial_stage_time/e2e_time"),
            ("model_switch_overhead_ms", "多模型数据流无缝切换", "vision→project 切换耗时"),
            ("feature_projection_transfer_gbps", "特征投影张量传输开销", "feature_bytes/latency"),
            ("multitask_isolation_overhead_ratio", "多任务资源隔离机制", "dual_stream/e2e - 1"),
        ],
    },
    "sd-unet-sampling": {
        "title": "SD UNet 迭代采样推理",
        "summary": "多步前向细碎卷积算子负载",
        "metrics_table": [
            ("loop_launch_overhead_ratio", "循环多次推理算子启动开销", "(loop - steps*step)/loop"),
            ("conv_array_utilization", "卷积脉动阵列利用率", "achieved_macs/peak_flops"),
            ("intermediate_tensor_cache_reuse_gbps", "中间张量片上缓存复用", "intermediate_bytes/latency"),
            ("latency_per_step_ms", "功耗与时延折中表现", "每步平均时延 proxy"),
        ],
    },
    "whisper-streaming": {
        "title": "Whisper 流式长音频推理",
        "summary": "时序滑动窗口超长序列计算",
        "metrics_table": [
            ("sliding_window_segment_throughput_fps", "滑动窗口张量分段处理能力", "windows/sec"),
            ("context_memory_growth_mb", "超长上下文显存占用控制", "context_bytes"),
            ("temporal_attention_latency_ms", "时序类 Attention 硬件延迟", "单窗口 attention 延迟"),
            ("streaming_io_throughput_gbps", "流式数据 IO 吞吐", "mel_bytes/latency"),
        ],
    },
    "int8-fp8-quant-inference": {
        "title": "INT8/FP8 量化推理",
        "summary": "低精度算子与数值稳定性验证",
        "metrics_table": [
            ("throughput_samples_per_s", "INT8/FP8 推理算力峰值", "batch/latency"),
            ("quant_dequant_overhead_ratio", "量化反量化硬件开销", "(quant_latency-fp32)/quant"),
            ("zero_point_offset_ops_ratio", "零值点偏移运算单元", "INT8 zero-point 路径占比 proxy"),
            ("low_precision_error_tolerance", "低精度下精度误差容错能力", "1 - max_relative_error"),
        ],
    },
}


def write_readme(bench_id, info):
    rows = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in info["metrics_table"])
    text = f"""# {info['title']}

> {info['summary']}

## 所属分类

[ai-inference](../README.md)

## 硬件考察维度

| 字段 | 指标 | 采集方式 |
|------|------|----------|
{rows}

## 快速运行

```bash
cd ai-inference/{bench_id}
pip install -r requirements.txt
python src/run.py
```

结果输出至 `results/benchmark_report.json`。

## 目录结构

```
{bench_id}/
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
"""
    (ROOT / bench_id / "README.md").write_text(text, encoding="utf-8")

    yaml = f"""id: {bench_id}
name: {info['title']}
category: ai-inference
summary: {info['summary']}
status: implemented
entrypoint: src/run.py
requirements: requirements.txt
output: results/benchmark_report.json
metrics:
"""
    for _, m, _ in info["metrics_table"]:
        yaml += f"  - {m}\n"
    yaml += "tags:\n  - chip-benchmark\n  - architecture-validation\n"
    (ROOT / bench_id / "benchmark.yaml").write_text(yaml, encoding="utf-8")


def main():
    for bench_id, info in BENCHES.items():
        write_readme(bench_id, info)
    print("Updated", len(BENCHES), "benchmark docs")


if __name__ == "__main__":
    main()
