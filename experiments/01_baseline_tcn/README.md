# 01 Baseline TCN

使用当前默认配置训练TCN模型的基线实验。

## 训练单个模型

```bash
python scripts/01_train.py --config ../../configs/models/tcn.yaml
```

## LOSO交叉验证

```bash
python scripts/03_train_loso.py --config ../../configs/edgeai_loso.yaml
```

## 配置

模型配置: `../../configs/models/tcn.yaml`

数据配置: `../../configs/edgeai.base.yaml`

## 输出

- `runs/` - 训练输出 (模型权重, 训练历史)
