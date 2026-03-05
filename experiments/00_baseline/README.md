# Experiment 00: Baseline Model

## 实验目的

建立正式的baseline模型，作为所有后续实验的性能参考标准。

## 配置

- **Density核**: skewed_gaussian (left=20ms, right=120ms)
- **模型**: TCN (256ch, 8layers, 3.29M params)
- **训练**: 1000 epochs, pos_fraction=0.5
- **数据**: 使用02_density_kernel的预计算数据

## 运行

```bash
cd experiments/00_baseline
python scripts/01_train.py
```

## 预期结果

基于02_density_kernel的结果：
- Test Count MAE: ~0.20
- Test Count MAE (pos): ~0.60
- Test Count MAE (neg): ~0.08

## 输出

- `runs/baseline/` - 训练输出
- `runs/baseline/best.pt` - 最佳checkpoint
- `runs/baseline/test_results_test.json` - 测试结果
