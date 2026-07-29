# 科学计算 HPC Benchmark

CFD、分子动力学、FFT、稀疏求解等 HPC 负载，考察 FP64、访存局部性与 MPI 通信能力。

## Benchmark 列表

| 目录 | 算法 | 状态 |
|------|------|------|
| [openfoam-cfd](./openfoam-cfd/) | OpenFOAM CFD 流体求解 | implemented |
| [lammps-md](./lammps-md/) | LAMMPS 分子动力学仿真 | implemented |
| [npb-hpc](./npb-hpc/) | NPB 并行 HPC 基准 | implemented |
| [3d-fft](./3d-fft/) | 3D-FFT 大规模傅里叶变换 | implemented |
| [fea-sparse-solver](./fea-sparse-solver/) | FEA 有限元稀疏矩阵求解 | implemented |
| [climate-model](./climate-model/) | 大气环流气候模式迭代计算 | implemented |

## 统一运行方式

每个 benchmark 目录结构一致：

```
<benchmark>/
├── src/run.py
├── requirements.txt
└── results/benchmark_report.json
```

```bash
cd scientific-computing/<benchmark>
pip install -r requirements.txt
python src/run.py
```

公共工具见 [common/bench_utils.py](../../common/bench_utils.py)（设备选择、计时、吞吐计算）。

## 使用说明

1. 进入对应 benchmark 目录查看 `README.md` 与 `benchmark.yaml`
2. 在目标芯片/设备上运行 `src/run.py` 采集指标
3. 根据 `results/benchmark_report.json` 输出架构对标报告
