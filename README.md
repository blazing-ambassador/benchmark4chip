# Benchmark4Chip — 极简可运行负载 + 架构 proxy 指标

> 面向芯片架构验证的 benchmark 集合：5 大场景、30 项负载，与 [overview.md](./overview.md) 中的硬件考察维度一一对应。

---

## 为什么做这个项目（Motivation）

对照 MLPerf 等通用基准，我们在内部芯片验证里更关心 **「负载是否够前沿、能否指向硬件瓶颈、能否做边界压测、训推是否贯通」**，而不是单一跑分排名。

| 常见短板 | Benchmark4Chip 的取向 |
|----------|------------------------|
| 负载偏保守，前沿场景覆盖不足 | 按 **训 / 推 / 端 / 云 / HPC** 拆细 30 类代表性负载 |
| 跑分导向，难定位架构瓶颈 | 每项负载绑定 **overview 中的硬件考察维度**，输出可对比的 proxy 指标 |
| 约束过死，难做破坏性压测 | 支持 **batch / 序列 / 步数 / 并发 sweep**，观察饱和拐点 |
| 训推一体考核弱 | 训练、推理、云端 FP8 全链路等 **分场景 + 可串联** |

**指标性质说明**：当前实现以 PyTorch 极简模型 + 计时 + 公式换算为主，属于 **软件侧架构 proxy**，便于在 CPU/GPU 上统一口径；接入自研芯片时需替换算子后端并接入真实 PMU/带宽计数（见 [接入自研芯片](#接入自研芯片)）。

---

## 项目概览

```
benchmark4chip/
├── README.md                 # 本文件
├── overview.md               # 30 项负载清单与硬件考察维度（权威指标定义）
├── common/
│   ├── bench_utils.py        # 全仓库共享：设备、计时、饱和拐点等
│   └── benchmark-template/   # 新增 benchmark 模板
├── scripts/                  # 生成、规整、冒烟脚本
├── logs/                     # 搭建与实现过程记录
├── ai-training/              # 6 benchmarks
├── ai-inference/             # 6 benchmarks
├── edge-ai/                  # 6 benchmarks
├── cloud-ai/                 # 6 benchmarks
└── scientific-computing/     # 6 benchmarks
```

| 分类 | 目录 | 数量 | 典型场景 |
|------|------|------|----------|
| AI 训练 | [ai-training](./ai-training/) | 6 | LLM 预训练、MoE、SD UNet 训练、ViT-MAE、TP 模拟、FP8 QAT |
| AI 推理 | [ai-inference](./ai-inference/) | 6 | LLM 解码、多模态、扩散采样、Whisper 流式、量化推理、BERT 批量编码 |
| 端侧 AI | [edge-ai](./edge-ai/) | 6 | YOLO、MobileNet、端侧 LLM、INT4、多任务、轻量化 MAE |
| 云侧 AI | [cloud-ai](./cloud-ai/) | 6 | 分布式 TP、MoE 批量、多租户、CLIP 跑批、Embedding 高并发、训推 FP8 |
| 科学计算 | [scientific-computing](./scientific-computing/) | 6 | CFD、分子动力学、NPB、3D-FFT、稀疏 FEA、气候模式 |

完整算法名称与考察维度见 **[overview.md](./overview.md)**；各分类索引见对应目录下的 `README.md`。

---

## 快速开始

### 环境

- Python 3.7+（建议 3.8+）
- PyTorch ≥ 1.9.0（GPU 测试需 CUDA 版 PyTorch）

### 运行单个 benchmark

```bash
cd ai-inference/bert-batch-encoding
pip install -r requirements.txt
python src/run.py
```

常用参数：

```bash
python src/run.py --device auto          # 有 CUDA 则用 GPU
python src/run.py --device cuda
python src/run.py --warmup 5 --iterations 20
python src/run.py --peak-flops 1e12        # 标定理论算力，便于利用率类指标
python src/run.py --output results/my_run.json
```

结果默认写入 **`results/benchmark_report.json`**（含 `sweep_points` 与 `summary`）。

### 五类各抽一项冒烟

在仓库根目录：

```bash
python scripts/run_smoke_all.py
```

---

## 单个 Benchmark 目录规范

```
<category>/<benchmark-id>/
├── README.md              # 算法说明、指标表、运行方式
├── benchmark.yaml         # id、metrics、status、entrypoint
├── requirements.txt       # 通常为 torch>=1.9.0
├── results/               # 运行后生成（*.json 默认 gitignore）
└── src/
    ├── model.py           # 极简负载（PyTorch）
    ├── metrics.py         # 架构 proxy 指标与 summary
    └── run.py             # CLI 入口
```

新增 benchmark 可复制 [common/benchmark-template](./common/benchmark-template/)，或扩展 `scripts/generate_benchmarks.py` / `scripts/category_archetypes.py` 后重新生成。

---

## 指标如何产生（统一方法论）

1. **Warmup + 多次迭代**：`common/bench_utils.timed_call` 测平均延迟。  
2. **参数 Sweep**：batch、序列长度、采样步数、并发等，形成 `sweep_points`。  
3. **Proxy 换算**：各 benchmark 的 `metrics.py` 按 overview 维度计算利用率、带宽、饱和拐点、误差容忍等。  
4. **汇总**：`build_summary()` 写入 JSON，便于跨平台对比。

各 benchmark 的 **字段含义与公式** 见该目录 `README.md` 中的「硬件考察维度」表。

---

## 接入自研芯片

| 步骤 | 说明 |
|------|------|
| 保留接口 | 保持 `run.py` 参数、`benchmark_report.json` 结构不变 |
| 替换算子 | 在 `model.py` 中改为 SDK / 编译器生成的 kernel，或 ONNX → 芯片 runtime |
| 标定理论峰值 | 运行时使用 `--peak-flops`、`--onchip-cache-kb`（部分 benchmark）等 |
| 真实计数 | 将计时、字节量、PMU 数据注入 `metrics.py` 的计算输入，替代纯估算 |

RISC-V / NPU 等若仅提供 C API，可在 `src/` 增加薄封装，`run.py` 仍负责 sweep 与写 JSON。

---

## 脚本与维护

| 脚本 | 用途 |
|------|------|
| [scripts/generate_benchmarks.py](./scripts/generate_benchmarks.py) | 从定义生成目录骨架与 README/yaml |
| [scripts/category_archetypes.py](./scripts/category_archetypes.py) | 24 类训练/端/云/HPC 的 model/metrics/run 模板 |
| [scripts/generate_four_categories.py](./scripts/generate_four_categories.py) | 批量写入四类 benchmark 实现 |
| [scripts/normalize_project.py](./scripts/normalize_project.py) | 统一 `common` 路径、清理冗余拷贝 |
| [scripts/run_smoke_all.py](./scripts/run_smoke_all.py) | 五类冒烟测试 |
| [scripts/setup_category_common.py](./scripts/setup_category_common.py) | 可选：将 root `bench_utils` 同步到分类目录 |

详见 [scripts/README.md](./scripts/README.md)。

---

## 实现状态

| 分类 | 可运行 `src/` | 说明 |
|------|----------------|------|
| ai-inference | 6/6 | 含手写优化的 BERT 等 |
| ai-training | 6/6 | 由 archetype 生成，支持 forward/backward 类 proxy |
| edge-ai | 6/6 | 端侧卷积 / LLM / 多任务等 |
| cloud-ai | 6/6 | 分布式与混部类 proxy |
| scientific-computing | 6/6 | FP64、FFT、SpMV、 stencil 等 |

**架构对标报告**（与竞品 / 上一代芯片对比的正式文档）需在业务侧基于 JSON 结果撰写，benchmark 内 checklist 仍为可选后续项。

---

## 日志与历史

过程记录见 [logs/](./logs/)（文件夹创建、ai-inference 实现、四类扩展等）。索引见 [logs/README.md](./logs/README.md)。

---

## 引用与扩展

- 负载与硬件维度权威列表：**[overview.md](./overview.md)**  
- 分类导航：**各目录 `README.md`**  
- 问题与扩展：优先改 `overview.md` 与对应 benchmark 的 `benchmark.yaml`，再同步 `scripts/` 中的生成定义，避免手改 30 份重复逻辑。
