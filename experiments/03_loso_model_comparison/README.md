# Experiment 03: LOSO Model Comparison

使用最佳 density 核配置，对比第5章所需的 0.7M 级咳嗽计数结构，并围绕 `TCN` / `UniGRU` / `BiGRU` 做结构消融。

## 目的

通过 Leave-One-Subject-Out (LOSO) 交叉验证完成两组证据：

1. `CNN1D / DSCNN / ResCNN / CRNN / BiCRNN / TCN / TCN-Attn / TCN+BiGRU` 的 0.7M 级结构横向对比。
2. `BiGRU-only / TCN-only / TCN+UniGRU / TCN+BiGRU` 的 0.7M 级消融实验。

## 实验流程

```
01_precompute.py → 06_ch5_rerun_queue.py → report
```

四张 4090 远端运行说明见：

```text
RERUN_0P7M_REMOTE_GUIDE.md
```

### Step 1: 预计算密度图 (01_precompute.py)

使用从02实验确定的最佳density核配置生成密度图。

```bash
python scripts/01_precompute.py
```

**输入**: 原始wav文件 + 咳嗽标签
**输出**: `data/` 目录 (S.npy, t.npy, density.npy)

### Step 2: 训练单模型 (02_train.py)

可选：训练单个模型用于快速验证。

```bash
# 训练所有模型
python scripts/02_train.py

# 只训练特定模型
python scripts/02_train.py --model-id M1
```

**输入**: `data/`
**输出**: `runs/{M1,M2,M3}/` (best.pt, history.json, etc.)

### Step 3: 第5章重跑队列 (06_ch5_rerun_queue.py)

正式重跑使用队列脚本统一调度 `03_loso.py`：

```bash
# 审计配置、模型ID和参数量
python scripts/06_ch5_rerun_queue.py audit

# 四张卡分别启动四个 shard
python scripts/06_ch5_rerun_queue.py run --shard-index 0 --num-shards 4 --device cuda:0
python scripts/06_ch5_rerun_queue.py run --shard-index 1 --num-shards 4 --device cuda:1
python scripts/06_ch5_rerun_queue.py run --shard-index 2 --num-shards 4 --device cuda:2
python scripts/06_ch5_rerun_queue.py run --shard-index 3 --num-shards 4 --device cuda:3

# 全部结束后汇总
python scripts/06_ch5_rerun_queue.py report
```

**输入**: `data/`
**输出**: `runs/structure_compare_v2_0p7m_<timestamp>/`、`runs/ablation_tcn_bigru_v2_0p7m_<timestamp>/` 和 `result/ch5_rerun_0p7m_<timestamp>/`

### 兼容入口：单模型 LOSO (03_loso.py)

对启用的模型运行 15-fold LOSO 交叉验证。默认启用列表来自 `experiment.yaml` 的 `loso.models`，当前为 `gru`、`tcn`、`tcn_gru`。

```bash
# 对默认三组消融模型运行LOSO，默认读 experiment.yaml 中的 500 epochs
python scripts/03_loso.py --device cuda

# 只对特定模型运行LOSO
python scripts/03_loso.py --model-id M2 --device cuda

# 快速烟测：只跑 tcn_gru 的 1 个 fold、1 个 epoch
python scripts/03_loso.py --model-id M2 --device cuda --epochs 1 --max-folds 1 --batch-size 8 --num-workers 0
```

**输入**: `data/`
**输出**: `runs/loso_model_compare_<timestamp>/{M0,M1,M2}/`

每个模型生成：
- `fold_XX_test_<subject>/` - 每个fold的训练结果
- `fold_XX_test_<subject>/best.pt` - 按 `val_count_mae` 保存的最佳权重
- `fold_XX_test_<subject>/test_results.json` - 使用 `best.pt` 在 left-out subject 上测试的结果
- `loso_summary.json` - 该模型的LOSO汇总统计

### Step 4: 整理论文结果 (05_thesis_ablation_report.py)

生成消融实验汇总表、fold 明细和可放入论文的 Markdown 草稿。

```bash
python scripts/05_thesis_ablation_report.py
```

**输入**: `runs/loso_model_compare_<timestamp>/`
**输出**: `result/thesis_ablation_<timestamp>/`

## 模型配置

| Model ID | Name | Architecture | Parameters |
|----------|------|--------------|------------|
| M0 | gru | GRU-only temporal baseline |
| M1 | tcn | TCN |
| M2 | tcn_gru | TCN + GRU |

当前 `experiment.yaml` 只保留论文消融所需的三组模型，避免误跑旧的 8 模型横评。

## 评估指标

- **Primary**: `test_count_mae` - 整体计数MAE
- **Secondary**: `test_count_mae_pos`, `test_count_mae_neg` - 正样本和负样本计数MAE

## 预期结果

形成论文第 5 章使用的消融证据：`TCN+GRU` 相比 `TCN` 和 `GRU` 的计数误差变化。

当前严格消融参数量：

| Model ID | Name | Trainable Params | FP32 Size |
|----------|------|-----------------:|----------:|
| M0 | gru | 115,841 | 0.442 MB |
| M1 | tcn | 609,665 | 2.326 MB |
| M2 | tcn_gru | 708,737 | 2.704 MB |

`M1` 是与 `M2` 中 TCN 前端匹配的 TCN-only 对照，不是旧配置中的大容量 TCN。TCN+GRU 参数量不等于独立 TCN 与独立 GRU 简单相加，因为各模型的输入投影和输出头不同。

## 从零跑通的最短命令

在仓库根目录执行：

```powershell
uv sync
.venv\Scripts\python.exe experiments\00_data_prep\scripts\01_download.py
.venv\Scripts\python.exe experiments\00_data_prep\scripts\02_build_manifest.py
.venv\Scripts\python.exe experiments\00_data_prep\scripts\03_split_subjects.py
```

然后进入本实验目录：

```powershell
cd experiments\03_loso_model_comparison
..\..\.venv\Scripts\python.exe scripts\01_precompute.py
..\..\.venv\Scripts\python.exe scripts\03_loso.py --model-id M2 --device cuda --epochs 1 --max-folds 1 --batch-size 8 --num-workers 0
```

确认 smoke test 成功后，再跑论文实验：

```powershell
..\..\.venv\Scripts\python.exe scripts\03_loso.py --model-id M2 --device cuda
..\..\.venv\Scripts\python.exe scripts\03_loso.py --model-id M1 --device cuda
..\..\.venv\Scripts\python.exe scripts\03_loso.py --model-id M0 --device cuda
..\..\.venv\Scripts\python.exe scripts\05_thesis_ablation_report.py
```

若时间充足，再另开实验比较旧的 `CRNN/BiCRNN/TCN-Attn/DSCNN/ResCNN/CNN1D`，但它们不是当前论文主线必需结果。

## 配置说明

在运行此实验前，需要更新`experiment.yaml`中的`density`配置，使用02实验中最佳的核函数配置：

```yaml
density:
  kernel: "skewed_gaussian"
  sigma_left_sec: 0.03  # 更新为最佳值
  sigma_right_sec: 0.10  # 更新为最佳值
  data_dir: "skewed_l30ms_r100ms"  # 更新为最佳目录
```
