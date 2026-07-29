# 端侧 AI Benchmark

NPU / 嵌入式 AI 芯片 / MCU 场景，考察低功耗、小内存、量化与多任务调度能力。

## Benchmark 列表

| 目录 | 算法 | 状态 |
|------|------|------|
| [yolov8-nano](./yolov8-nano/) | YOLOv8-Nano 实时检测推理 | implemented |
| [mobilenetv3](./mobilenetv3/) | MobileNetV3 分类推理 | implemented |
| [distilbert-edge](./distilbert-edge/) | DistilBERT 端侧小 LLM 推理 | implemented |
| [qwen-int4](./qwen-int4/) | Qwen-1.8B INT4 量化离线推理 | implemented |
| [mediapipe-multitask](./mediapipe-multitask/) | MediaPipe 多任务串行推理 | implemented |
| [lightweight-mae](./lightweight-mae/) | 轻量化 MAE 视觉提取推理 | implemented |

## 统一运行方式

每个 benchmark 目录结构一致：

```
<benchmark>/
├── src/run.py
├── requirements.txt
└── results/benchmark_report.json
```

```bash
cd edge-ai/<benchmark>
pip install -r requirements.txt
python src/run.py
```

公共工具见 [common/bench_utils.py](../../common/bench_utils.py)（设备选择、计时、吞吐计算）。

## 使用说明

1. 进入对应 benchmark 目录查看 `README.md` 与 `benchmark.yaml`
2. 在目标芯片/设备上运行 `src/run.py` 采集指标
3. 根据 `results/benchmark_report.json` 输出架构对标报告
