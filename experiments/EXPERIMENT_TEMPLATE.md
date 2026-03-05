# 实验报告模板

## 实验基本信息

- **实验ID**: EXP_XX_YYYYMMDD
- **实验名称**: 简短描述
- **负责人**: [你的名字]
- **开始日期**: YYYY-MM-DD
- **结束日期**: YYYY-MM-DD (进行中/已完成)

---

## 1. 实验目的

**明确陈述这个实验要解决的问题或验证的假设**

示例:
- 验证非对称density核是否能改善cough counting性能
- 对比TCN、TCN_GRU、CRNN三种模型的泛化能力

---

## 2. 背景/Motivation

**为什么做这个实验？基于什么观察？**

示例:
- 之前的实验显示使用对称高斯核时，模型在cough事件边界预测不够准确
- 假设：cough事件的起始和结束应该有不同的时间衰减特性

---

## 3. 假设 (Hypothesis)

**可验证的假设陈述**

示例:
- H1: 使用左窄右宽的非对称核可以降低Test MAE
- H2: sigma_left=20ms, sigma_right=120ms是最佳配置

---

## 4. 实验设计

### 4.1 变量

| 变量类型 | 变量名 | 值/范围 |
|----------|--------|---------|
| 自变量 | density核类型 | gaussian, skewed_gaussian, cosine |
| 自变量 | sigma_left | 10ms, 20ms, 30ms |
| 自变量 | sigma_right | 80ms, 100ms, 120ms |
| 因变量 | Test Count MAE | - |
| 控制变量 | 模型架构 | TCN (256ch, 8layers) |
| 控制变量 | pos_fraction | 0.5 |
| 控制变量 | epochs | 500 |

### 4.2 对比基线 (Baseline)

**明确说明基线是什么**

示例:
- 基线1: 对称高斯核 (sigma=50ms), Test MAE = X.XX
- 基线2: 之前最佳模型 (skewed_30_100), Test MAE = 0.48

### 4.3 评估指标

| 指标 | 重要性 | 说明 |
|------|--------|------|
| Test Count MAE | ★★★ | 主要指标 |
| Test Count MAE (pos) | ★★ | Positive样本性能 |
| Val Count MAE | ★ | 验证集性能 |

---

## 5. 实验配置

### 5.1 数据配置
- Split: train/val/test
- 正负样本比例

### 5.2 模型配置
- 架构
- 参数量
- 优化器
- 学习率

### 5.3 训练配置
- Epochs
- Batch size
- Loss weights
- 其他超参数

---

## 6. 实验结果

### 6.1 主要结果

| 配置 | Val MAE | Test MAE | Test Pos MAE | 变化 |
|------|---------|----------|--------------|------|
| 基线 | X.XX | X.XX | X.XX | - |
| 配置1 | X.XX | X.XX | X.XX | ▲X.XX% |
| 配置2 | X.XX | X.XX | X.XX | ▼X.XX% |

### 6.2 可视化

[插入图表]

### 6.3 详细分析

- 哪些配置有效？为什么？
- 哪些配置无效？为什么？
- 有意外的发现吗？

---

## 7. 结论

### 7.1 假设验证

| 假设 | 结果 | 说明 |
|------|------|------|
| H1 | ✓/✗ | [说明] |
| H2 | ✓/✗ | [说明] |

### 7.2 最佳配置

- **最佳配置**: [配置名称]
- **性能**: Test MAE = X.XX
- **vs基线**: 提升/下降 X.XX%

### 7.3 下一步

基于本实验结果，下一步应该：
1. [具体建议]
2. [具体建议]

---

## 8. 记录与可复现性

### 8.1 文件位置
- 实验目录: `experiments/XX_name/`
- 配置文件: `experiment.yaml`
- 结果文件: `result/summary_*.json`

### 8.2 运行命令
```bash
cd experiments/XX_name
python scripts/01_preprocess.py
python scripts/02_train.py
python scripts/03_evaluate.py
```

### 8.3 环境信息
- Python版本: X.X.X
- PyTorch版本: X.X.X
- CUDA版本: X.X

---

## 9. 附录

### 9.1 详细数据
[附加表格或数据]

### 9.2 错误日志
[如果有训练失败或异常]
