# Experiment 02: Density Kernel Comparison

## 实验目的

**验证假设**: 使用非对称的左窄右宽(skewed)密度核可以改善cough counting性能

### 背景

Cough事件具有明显的时间不对称性：
- **起始**: 突然、清晰，应该用较窄的核
- **结束**: 渐弱、模糊，应该用较宽的核

对称的高斯核无法捕捉这种特性，可能导致：
- 边界预测不准确
- 密度图过于平滑或尖锐

### 假设

| 假设 | 说明 |
|------|------|
| H1 | 左窄右宽的skewed gaussian核优于对称gaussian核 |
| H2 | 存在最佳的(sigma_left, sigma_right)组合 |
| H3 | sigma_left约20-30ms, sigma_right约100-150ms为最佳区间 |

---

## 实验设计

### 变量

| 变量类型 | 变量名 | 值 |
|----------|--------|-----|
| **自变量** | density核类型 | gaussian, skewed_gaussian, cosine |
| **自变量** | sigma_left | 15ms, 20ms, 25ms, 30ms, 40ms |
| **自变量** | sigma_right | 80ms, 100ms, 120ms, 150ms |
| **因变量** | Test Count MAE | 主要评估指标 |
| **因变量** | Test Count MAE (pos) | Positive样本性能 |
| **控制变量** | 模型 | TCN (256ch, 8layers, 3.29M) |
| **控制变量** | pos_fraction | 0.5 (平衡采样) |
| **控制变量** | epochs | 500 |
| **控制变量** | 动态loss balancer | 启用 |

### Baseline

| 配置 | Test MAE | 来源 |
|------|----------|------|
| 对称高斯 (sigma=50ms) | ~0.5+ (待确认) | 预估 |
| skewed_30_100 | 0.48 | 本实验1E |

---

## 核函数配置

### Gaussian (对称)

| ID | Name | sigma | 描述 |
|----|------|-------|------|
| 1A | gaussian_50ms | 50ms | 标准配置 |
| 1B | gaussian_100ms | 100ms | 较宽 |
| 1C | gaussian_150ms | 150ms | 很宽 |

### Skewed Gaussian (左窄右宽)

| ID | Name | sigma_left | sigma_right | Ratio | 描述 |
|----|------|------------|-------------|-------|------|
| 1D | skewed_40_80 | 40ms | 80ms | 1:2 | 轻微不对称 |
| 1E | skewed_30_100 | 30ms | 100ms | 1:3.3 | **当前最佳** |
| 1F | skewed_20_120 | 20ms | 120ms | 1:6 | 更不对称 |
| 1G | skewed_20_120 | 20ms | 120ms | 1:6 | 重跑(pos=0.5) |
| 1H | skewed_25_150 | 25ms | 150ms | 1:6 | 更宽右尾 |
| 1I | skewed_15_120 | 15ms | 120ms | 1:8 | 极左窄 |
| 1J | skewed_20_150 | 20ms | 150ms | 1:7.5 | 极右宽 |

### Cosine

| ID | Name | half_width | 描述 |
|----|------|------------|------|
| - | cosine_100ms | 100ms | 余弦核 |

---

## 实验流程

```
01_precompute.py → 02_train.py → 03_evaluate.py
```

### Step 1: 预计算密度图
```bash
python scripts/01_precompute.py --kernel-ids 1E 1G 1H 1I 1J
```

### Step 2: 训练模型
```bash
# 训练所有核
python scripts/02_train.py

# 训练特定核
python scripts/02_train.py --kernel-id 1E --epochs 500

# 阶段训练: 用不同pos_fraction继续训练
python scripts/02_train.py --kernel-id 1E --epochs 1000 --pos-fraction 0.3
```

### Step 3: 整理结果
```bash
python scripts/03_evaluate.py
```

---

## 当前结果 (进行中)

| Kernel | sigma_left | sigma_right | Best Epoch | Val MAE | Test MAE | 状态 |
|--------|------------|-------------|------------|---------|----------|------|
| 1E | 30ms | 100ms | 241 | 0.81 | **0.48** | ✓ |
| 1H | 25ms | 150ms | 341 | 0.81 | ? | 训练中 |
| 1I | 15ms | 120ms | 218 | 0.82 | ? | 训练中 |
| 1J | 20ms | 150ms | 357 | 0.97 | ? | 训练中 |
| 1G | 20ms | 120ms | 248 | 1.01 | ? | 训练中 |

**初步结论**:
- 1H和1I的Val MAE与1E非常接近 (0.81-0.82)
- 有望在Test集上超越1E的0.48

---

## 预期结果

1. **验证H1**: Skewed核是否优于对称核
2. **验证H2**: 找到最佳(sigma_left, sigma_right)组合
3. **输出**: 最佳density核配置，用于03实验的LOSO评估

---

## 输出文件

```
runs/
├── {kernel_id}/
│   ├── best.pt                  # 最佳checkpoint (用于测试)
│   ├── best_info.json           # 最佳epoch的详细信息
│   ├── history.json             # 完整训练历史
│   ├── train_summary.json       # 训练汇总
│   ├── config.yaml              # 配置快照
│   └── test_results_test.json   # 测试集结果

result/
├── kernel_comparison_summary_{timestamp}.json  # 数值结果
└── kernel_comparison_report_{timestamp}.md     # 实验报告
```

---

## 实验报告

训练完成后，将生成详细的实验报告，包含：
- 所有配置的完整对比
- 可视化图表
- 假设验证结果
- 最佳配置推荐
- 下一步实验建议
