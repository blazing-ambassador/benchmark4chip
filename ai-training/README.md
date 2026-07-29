# AI 训练 Benchmark

通用 LLM、扩散模型、视觉 Transformer 等训练负载，考察稠密/稀疏算力、并行通信与混合精度能力。

## Benchmark 列表

| 目录 | 算法 | 状态 |
|------|------|------|
| [llama-7b-pretrain](./llama-7b-pretrain/) | LLaMA-7B 稠密预训练 | implemented |
| [moe-training](./moe-training/) | MoE 混合专家模型训练 | implemented |
| [sd-unet-training](./sd-unet-training/) | Stable Diffusion UNet 完整训练 | implemented |
| [vit-mae-training](./vit-mae-training/) | ViT-MAE 视觉自监督训练 | implemented |
| [megatron-tp-simulation](./megatron-tp-simulation/) | Megatron 超大模型张量并行模拟 | implemented |
| [fp8-qat-training](./fp8-qat-training/) | FP8 QAT 量化感知训练 | implemented |

## 统一运行方式

每个 benchmark 目录结构一致：

```
<benchmark>/
├── src/run.py
├── requirements.txt
└── results/benchmark_report.json
```

```bash
cd ai-training/<benchmark>
pip install -r requirements.txt
python src/run.py
```

公共工具见 [common/bench_utils.py](../../common/bench_utils.py)（设备选择、计时、吞吐计算）。

## 使用说明

1. 进入对应 benchmark 目录查看 `README.md` 与 `benchmark.yaml`
2. 在目标芯片/设备上运行 `src/run.py` 采集指标
3. 根据 `results/benchmark_report.json` 输出架构对标报告
