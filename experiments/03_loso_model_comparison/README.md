# Experiment 03: LOSO Model Comparison

使用最佳density核配置，对比TCN/TCN_GRU/CRNN三模型的LOSO性能。

## 目的

通过Leave-One-Subject-Out (LOSO)交叉验证，找出泛化能力最好的模型架构。

## 实验流程

```
01_precompute.py → 02_train.py → 03_loso.py → 04_evaluate.py
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

### Step 3: LOSO评估 (03_loso.py)

对每个模型运行15-fold LOSO交叉验证。

```bash
# 对所有模型运行LOSO
python scripts/03_loso.py

# 只对特定模型运行LOSO
python scripts/03_loso.py --model-id M1
```

**输入**: `data/`
**输出**: `runs/loso_model_compare_<timestamp>/{M1,M2,M3}/`

每个模型生成：
- `fold_XX_test_<subject>/` - 每个fold的训练结果
- `loso_summary.json` - 该模型的LOSO汇总统计

### Step 4: 整理结果 (04_evaluate.py)

生成汇总报告并保存到`result/`目录。

```bash
python scripts/04_evaluate.py
```

**输入**: `runs/loso_model_compare_<timestamp>/`
**输出**: `result/loso_summary_<timestamp>.json`, `result/loso_report_<timestamp>.md`

## 模型配置

| Model ID | Name | Architecture | Parameters |
|----------|------|--------------|------------|
| M1 | tcn | TCN (256 ch, 8 layers) | ~3.3M |
| M2 | tcn_gru | TCN (128 ch, 6 layers) + BiGRU (128) | ~3.5M |
| M3 | crnn | Conv1D (64/128/256) + BiGRU (128, 2 layers) | ~2.5M |

## 评估指标

- **Primary**: `test_count_mae_pos` - positive样本的计数MAE
- **Secondary**: `test_count_mae` - 整体计数MAE

## 预期结果

找到泛化能力最好的模型，用于实验04的最终训练。

## 配置说明

在运行此实验前，需要更新`experiment.yaml`中的`density`配置，使用02实验中最佳的核函数配置：

```yaml
density:
  kernel: "skewed_gaussian"
  sigma_left_sec: 0.03  # 更新为最佳值
  sigma_right_sec: 0.10  # 更新为最佳值
  data_dir: "skewed_l30ms_r100ms"  # 更新为最佳目录
```
