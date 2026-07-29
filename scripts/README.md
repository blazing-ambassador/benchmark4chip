# 脚本说明

| 脚本 | 说明 |
|------|------|
| `generate_benchmarks.py` | 根据 `CATEGORIES` 生成五类目录的 README / benchmark.yaml 骨架 |
| `category_archetypes.py` | 训练、端侧、云、HPC 共 24 个 benchmark 的 model/metrics/run 模板 |
| `generate_four_categories.py` | 写入 ai-training、edge-ai、cloud-ai、scientific-computing 的可运行 `src/` |
| `setup_category_common.py` | 将根目录 `common/bench_utils.py` 同步到各分类 `common/`（可选） |
| `normalize_project.py` | 统一 `run.py` 中 `common` 路径为仓库根 `common/`，清理冗余拷贝 |
| `run_smoke_all.py` | 五类各 1 个 benchmark 快速冒烟 |

## 推荐工作流

**新增/重生成四类 benchmark 实现：**

```bash
python scripts/setup_category_common.py   # 可选
python scripts/generate_four_categories.py
python scripts/normalize_project.py
```

**仅更新目录骨架（无 src）：**

```bash
python scripts/generate_benchmarks.py
```

**验证：**

```bash
python scripts/run_smoke_all.py
```
