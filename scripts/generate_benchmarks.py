#!/usr/bin/env python3
"""Generate benchmark scaffold from overview definitions."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CATEGORIES = {
    "ai-training": {
        "title": "AI 训练 Benchmark",
        "description": "通用 LLM、扩散模型、视觉 Transformer 等训练负载，考察稠密/稀疏算力、并行通信与混合精度能力。",
        "benchmarks": [
            {
                "id": "llama-7b-pretrain",
                "name": "LLaMA-7B 稠密预训练",
                "summary": "通用 LLM 稠密算力 + 并行通信",
                "metrics": [
                    "BF16/FP16 张量算力利用率",
                    "片上 SRAM 缓存命中率",
                    "激活重计算显存开销",
                    "多卡 TP/PP 集体通信带宽",
                    "NOC 片上网络吞吐",
                    "显存带宽饱和能力",
                ],
            },
            {
                "id": "moe-training",
                "name": "MoE 混合专家模型训练",
                "summary": "稀疏动态算力",
                "metrics": [
                    "动态不规则张量寻址",
                    "路由算子硬件加速",
                    "动态显存分配效率",
                    "专家负载均衡调度能力",
                    "稀疏矩阵计算单元效能",
                    "跨专家数据搬运带宽",
                ],
            },
            {
                "id": "sd-unet-training",
                "name": "Stable Diffusion UNet 完整训练",
                "summary": "卷积 + 循环迭代细碎算子",
                "metrics": [
                    "小尺寸卷积算子执行效率",
                    "算子调度流水线开销",
                    "残差分支数据流复用",
                    "大量小 Tensor 频繁访存",
                    "低精度数值稳定性",
                ],
            },
            {
                "id": "vit-mae-training",
                "name": "ViT-MAE 视觉自监督训练",
                "summary": "纯视觉 Transformer",
                "metrics": [
                    "长序列注意力矩阵乘算力",
                    "Patch 张量重排开销",
                    "大维度张量片内搬运",
                    "归一化层硬件加速",
                    "高带宽 DRAM 吞吐",
                ],
            },
            {
                "id": "megatron-tp-simulation",
                "name": "Megatron 超大模型张量并行模拟",
                "summary": "多芯片互联",
                "metrics": [
                    "MCM 多芯片封装高速接口带宽",
                    "AllReduce/AllGather 规约延迟",
                    "多 die NOC 拓扑效率",
                    "参数分片流水并行能力",
                    "跨芯片同步开销",
                ],
            },
            {
                "id": "fp8-qat-training",
                "name": "FP8 QAT 量化感知训练",
                "summary": "混合精度硬件适配",
                "metrics": [
                    "FP8 训练原生算力",
                    "量化/伪量化算子硬加速",
                    "浮点与低精度数据流切换开销",
                    "量化误差硬件数值保真度",
                    "混合精度存储带宽利用率",
                ],
            },
        ],
    },
    "ai-inference": {
        "title": "AI 推理 Benchmark",
        "description": "LLM 解码、多模态、扩散采样、流式音频等推理负载，考察 KV Cache、流水线与低精度推理能力。",
        "benchmarks": [
            {
                "id": "llama-7b-decode",
                "name": "LLaMA-7B 增量解码推理",
                "summary": "KV 缓存复用、自注意力增量计算",
                "metrics": [
                    "片上高速缓存对 KV Cache 驻留能力",
                    "动态序列长度算力弹性",
                    "解码时延/首包响应延迟",
                    "批量推理调度单元",
                    "权重带宽瓶颈",
                ],
            },
            {
                "id": "llava-multimodal",
                "name": "Llava 多模态图文联合推理",
                "summary": "视觉编码器 + LLM 异构流水线",
                "metrics": [
                    "异构算子流水线并行执行能力",
                    "多模型数据流无缝切换",
                    "特征投影张量传输开销",
                    "多任务资源隔离机制",
                ],
            },
            {
                "id": "sd-unet-sampling",
                "name": "SD UNet 迭代采样推理",
                "summary": "多步前向细碎卷积算子负载",
                "metrics": [
                    "循环多次推理算子启动开销",
                    "卷积脉动阵列利用率",
                    "中间张量片上缓存复用",
                    "功耗与时延折中表现",
                ],
            },
            {
                "id": "whisper-streaming",
                "name": "Whisper 流式长音频推理",
                "summary": "时序滑动窗口超长序列计算",
                "metrics": [
                    "滑动窗口张量分段处理能力",
                    "超长上下文显存占用控制",
                    "时序类 Attention 硬件延迟",
                    "流式数据 IO 吞吐",
                ],
            },
            {
                "id": "int8-fp8-quant-inference",
                "name": "INT8/FP8 量化推理",
                "summary": "低精度算子与数值稳定性验证",
                "metrics": [
                    "INT8/FP8 推理算力峰值",
                    "量化反量化硬件开销",
                    "零值点偏移运算单元",
                    "低精度下精度误差容错能力",
                ],
            },
            {
                "id": "bert-batch-encoding",
                "name": "BERT 批量向量编码推理",
                "summary": "小 Transformer 高并发吞吐压力",
                "metrics": [
                    "批量并行计算单元利用率",
                    "短序列算子调度效率",
                    "高并发下算力溢出阈值",
                    "片上缓存批量数据吞吐",
                ],
            },
        ],
    },
    "edge-ai": {
        "title": "端侧 AI Benchmark",
        "description": "NPU / 嵌入式 AI 芯片 / MCU 场景，考察低功耗、小内存、量化与多任务调度能力。",
        "benchmarks": [
            {
                "id": "yolov8-nano",
                "name": "YOLOv8-Nano 实时检测推理",
                "summary": "轻量化卷积 + 后处理 NPU 算子",
                "metrics": [
                    "深度可分离卷积硬件效率",
                    "NMS/置信度筛选后处理硬加速",
                    "低功耗模式算力输出",
                    "片内小容量 SRAM 利用率",
                ],
            },
            {
                "id": "mobilenetv3",
                "name": "MobileNetV3 分类推理",
                "summary": "深度可分离卷积低带宽负载",
                "metrics": [
                    "卷积数据流寄存器复用",
                    "访存带宽节约架构",
                    "极低功耗下有效算力",
                    "MCU 级窄内存接口适配能力",
                ],
            },
            {
                "id": "distilbert-edge",
                "name": "DistilBERT 端侧小 LLM 推理",
                "summary": "片上存储极限约束测试",
                "metrics": [
                    "权重压缩后片上 Flash 加载速度",
                    "极小缓存下 Attention 分块计算",
                    "内存带宽受限场景算力衰减率",
                ],
            },
            {
                "id": "qwen-int4",
                "name": "Qwen-1.8B INT4 量化离线推理",
                "summary": "权重压缩与带宽瓶颈",
                "metrics": [
                    "INT4 解压硬件单元",
                    "4bit 权重访存带宽压力",
                    "片上缓存分块加载策略",
                    "低比特数值计算稳定性",
                ],
            },
            {
                "id": "mediapipe-multitask",
                "name": "MediaPipe 多任务串行推理",
                "summary": "多模型调度开销",
                "metrics": [
                    "模型快速切换上下文保存/恢复开销",
                    "流水线任务硬件调度器",
                    "多算子串行流水吞吐",
                    "功耗动态调节能力",
                ],
            },
            {
                "id": "lightweight-mae",
                "name": "轻量化 MAE 视觉提取推理",
                "summary": "端侧视觉特征低功耗负载",
                "metrics": [
                    "低帧率算力节流控制",
                    "图像 Patch 预处理硬件加速",
                    "休眠/运行功耗切换表现",
                ],
            },
        ],
    },
    "cloud-ai": {
        "title": "云端 AI Benchmark",
        "description": "数据中心大规模集群场景，考察多芯片互联、稀疏推理、租户隔离与训推一体能力。",
        "benchmarks": [
            {
                "id": "distributed-tp-inference",
                "name": "大模型分布式张量并行推理",
                "summary": "MCM 多芯片互联与规约通信",
                "metrics": [
                    "多 die 高速互联带宽",
                    "推理阶段 AllReduce 规约延迟",
                    "片间 NOC 拥塞控制",
                    "多芯片负载均衡调度",
                ],
            },
            {
                "id": "moe-cloud-batch",
                "name": "MoE 稀疏专家云端批量推理",
                "summary": "动态路由负载不均衡访存",
                "metrics": [
                    "稀疏 Token 动态分发硬件单元",
                    "专家核算力动态调度",
                    "不规则访存带宽冲突规避",
                    "批量任务负载打散能力",
                ],
            },
            {
                "id": "multi-model-tenant",
                "name": "多模型混部租户压测",
                "summary": "算力调度与资源隔离开销",
                "metrics": [
                    "硬件资源虚拟化隔离",
                    "算力切片抢占调度",
                    "缓存分区保护",
                    "多任务 QoS 时延保障能力",
                ],
            },
            {
                "id": "clip-distributed",
                "name": "CLIP 海量图片分布式特征抽取",
                "summary": "IO + 算力满负载跑批",
                "metrics": [
                    "PCIe/主机 DMA 带宽上限",
                    "批量图像预处理硬加速",
                    "算力持续满载稳定性",
                    "DRAM 持续读写带宽",
                ],
            },
            {
                "id": "embedding-concurrent",
                "name": "Embedding 高并发向量推理",
                "summary": "网络吞吐与批量调度压力",
                "metrics": [
                    "超大 Batch 并行算力",
                    "向量归一化硬件加速",
                    "网卡直连张量传输开销",
                    "并发请求队列硬件调度器",
                ],
            },
            {
                "id": "train-infer-fp8-e2e",
                "name": "训推一体 FP8 全链路负载",
                "summary": "混合精度端到端硬件适配",
                "metrics": [
                    "FP8 训推统一数据通路",
                    "训练量化→推理加载无缝流转",
                    "混合精度存储带宽综合利用率",
                ],
            },
        ],
    },
    "scientific-computing": {
        "title": "科学计算 HPC Benchmark",
        "description": "CFD、分子动力学、FFT、稀疏求解等 HPC 负载，考察 FP64、访存局部性与 MPI 通信能力。",
        "benchmarks": [
            {
                "id": "openfoam-cfd",
                "name": "OpenFOAM CFD 流体求解",
                "summary": "双精度稠密线性代数算力",
                "metrics": [
                    "FP64 双精度峰值算力",
                    "稠密矩阵乘阵列效率",
                    "迭代求解器访存局部性",
                    "大规模方程组存储带宽",
                ],
            },
            {
                "id": "lammps-md",
                "name": "LAMMPS 分子动力学仿真",
                "summary": "不规则随机内存访问",
                "metrics": [
                    "随机散列访存架构效率",
                    "粒子力场计算并行度",
                    "非连续地址访问缓存命中率",
                    "大数组寻址延迟",
                ],
            },
            {
                "id": "npb-hpc",
                "name": "NPB 并行 HPC 基准",
                "summary": "MPI 通信与浮点算力匹配度",
                "metrics": [
                    "多进程 MPI 通信接口带宽",
                    "计算/通信重叠执行能力",
                    "浮点运算单元持续效率",
                    "并行同步开销",
                ],
            },
            {
                "id": "3d-fft",
                "name": "3D-FFT 大规模傅里叶变换",
                "summary": "内存带宽极限压榨",
                "metrics": [
                    "多维张量转置硬件开销",
                    "跨步访存带宽利用率",
                    "FFT 蝶形运算单元加速",
                    "大尺寸数组缓存分块能力",
                ],
            },
            {
                "id": "fea-sparse-solver",
                "name": "FEA 有限元稀疏矩阵求解",
                "summary": "SpMV 稀疏张量运算",
                "metrics": [
                    "SpMV 稀疏矩阵乘法专用单元",
                    "不规则稀疏存储遍历效率",
                    "非零元素索引硬件寻址",
                    "带宽受限下算力衰减",
                ],
            },
            {
                "id": "climate-model",
                "name": "大气环流气候模式迭代计算",
                "summary": "大数组持续访存压力",
                "metrics": [
                    "超大型数组流式读写带宽",
                    "长时间步迭代算力稳定性",
                    "循环变量寄存器复用",
                    "大容量 DRAM 吞吐上限",
                ],
            },
        ],
    },
}


def yaml_list(items, indent=2):
    pad = " " * indent
    return "\n".join(f"{pad}- {item}" for item in items)


def write_benchmark(category: str, bench: dict) -> None:
    bench_dir = ROOT / category / bench["id"]
    bench_dir.mkdir(parents=True, exist_ok=True)

    readme = f"""# {bench['name']}

> {bench['summary']}

## 所属分类

[{category}](../README.md)

## 算法说明

本 benchmark 用于评估芯片在 **{bench['name']}** 场景下的硬件表现。

## 硬件考察维度

{chr(10).join(f'- {m}' for m in bench['metrics'])}

## 目录结构

```
{bench['id']}/
├── README.md
├── benchmark.yaml
└── src/           # 算法实现与运行脚本（待补充）
```

## 运行状态

- [ ] 负载实现
- [ ] 指标采集
- [ ] 架构对标报告
"""
    (bench_dir / "README.md").write_text(readme, encoding="utf-8")

    yaml_content = f"""id: {bench['id']}
name: {bench['name']}
category: {category}
summary: {bench['summary']}
status: planned
metrics:
{yaml_list(bench['metrics'], 2)}
tags:
  - chip-benchmark
  - architecture-validation
"""
    (bench_dir / "benchmark.yaml").write_text(yaml_content, encoding="utf-8")

    src_dir = bench_dir / "src"
    src_dir.mkdir(exist_ok=True)
    placeholder = src_dir / ".gitkeep"
    if not placeholder.exists():
        placeholder.write_text("", encoding="utf-8")


def write_category_readme(category: str, info: dict) -> None:
    cat_dir = ROOT / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    table_rows = "\n".join(
        f"| [{b['id']}](./{b['id']}/) | {b['name']} | {b['summary']} |"
        for b in info["benchmarks"]
    )

    readme = f"""# {info['title']}

{info['description']}

## Benchmark 列表

| 目录 | 算法 | 负载特征 |
|------|------|----------|
{table_rows}

## 使用说明

1. 进入对应 benchmark 目录查看 `README.md` 与 `benchmark.yaml`
2. 在 `src/` 下补充算法实现与运行脚本
3. 按硬件考察维度采集指标并输出架构分析报告
"""
    (cat_dir / "README.md").write_text(readme, encoding="utf-8")


def write_template() -> None:
    template_dir = ROOT / "common" / "benchmark-template"
    template_dir.mkdir(parents=True, exist_ok=True)

    (template_dir / "README.md").write_text(
        """# <Benchmark Name>

> <One-line summary>

## 所属分类

[<category>](../README.md)

## 算法说明

<!-- 描述算法、输入输出、典型规模 -->

## 硬件考察维度

- <!-- metric 1 -->
- <!-- metric 2 -->

## 目录结构

```
<benchmark-id>/
├── README.md
├── benchmark.yaml
└── src/
```

## 运行状态

- [ ] 负载实现
- [ ] 指标采集
- [ ] 架构对标报告
""",
        encoding="utf-8",
    )

    (template_dir / "benchmark.yaml").write_text(
        """id: <benchmark-id>
name: <Benchmark Name>
category: <category>
summary: <One-line summary>
status: planned
metrics:
  - <metric 1>
  - <metric 2>
tags:
  - chip-benchmark
  - architecture-validation
""",
        encoding="utf-8",
    )

    (template_dir / "src").mkdir(exist_ok=True)
    (template_dir / "src" / ".gitkeep").write_text("", encoding="utf-8")


def main() -> None:
    write_template()
    for category, info in CATEGORIES.items():
        write_category_readme(category, info)
        for bench in info["benchmarks"]:
            write_benchmark(category, bench)
    print(f"Generated {sum(len(v['benchmarks']) for v in CATEGORIES.values())} benchmarks in {len(CATEGORIES)} categories.")


if __name__ == "__main__":
    main()
