# AI 推理 Benchmark

LLM 解码、多模态、扩散采样、流式音频等推理负载，考察 KV Cache、流水线与低精度推理能力。

## Benchmark 列表

| 目录 | 算法 | 状态 |
|------|------|------|
| [llama-7b-decode](./llama-7b-decode/) | LLaMA-7B 增量解码推理 | implemented |
| [llava-multimodal](./llava-multimodal/) | Llava 多模态图文联合推理 | implemented |
| [sd-unet-sampling](./sd-unet-sampling/) | SD UNet 迭代采样推理 | implemented |
| [whisper-streaming](./whisper-streaming/) | Whisper 流式长音频推理 | implemented |
| [int8-fp8-quant-inference](./int8-fp8-quant-inference/) | INT8/FP8 量化推理 | implemented |
| [bert-batch-encoding](./bert-batch-encoding/) | BERT 批量向量编码推理 | implemented |

## 统一运行方式

每个 benchmark 目录结构一致：

```
<benchmark>/
├── src/run.py
├── requirements.txt
└── results/benchmark_report.json
```

```bash
cd ai-inference/<benchmark>
pip install -r requirements.txt
python src/run.py
```

公共工具见 [common/bench_utils.py](../../common/bench_utils.py)（设备选择、计时、吞吐计算）。

## 使用说明

1. 进入对应 benchmark 目录查看 `README.md` 与 `benchmark.yaml`
2. 在目标芯片/设备上运行 `src/run.py` 采集指标
3. 根据 `results/benchmark_report.json` 输出架构对标报告
