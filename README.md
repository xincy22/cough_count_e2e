# cough_count_e2e

基于 EdgeAI 咳嗽音频数据的端到端咳嗽次数估计。项目覆盖从数据下载、特征预计算、窗口采样到模型训练的完整流程。

## 主要流程
- 下载并校验 EdgeAI public_dataset.zip
- 生成 manifest.csv（包含事件时间、计数等元信息）
- 受试者级别划分 train/val/test
- 预计算 STFT log-magnitude 与密度标签，并按时间窗口切片
- 训练 CNN1D / TCN / CRNN，输出 runs/ 目录下的权重与历史记录

## 环境与安装
- Python >= 3.10
- 推荐 uv：

```bash
uv sync
```

- 或使用 pip：

```bash
python -m pip install -e .
```

## 数据准备
在仓库根目录依次执行：

```bash
python scripts/01_download_edgeai.py
python scripts/02_build_manifest_edgeai.py
python scripts/04_split_edgeai_subjects.py
python scripts/05_precompute_edgeai_windows.py --mic both
```

生成的数据位于 `data/` 下（默认会被 .gitignore 忽略）。

## 训练
使用默认配置训练：

```bash
python scripts/07_train_edgeai.py --config configs/edgeai.yaml
```

配置项可在 `configs/edgeai.yaml` 中调整，包括：
- `model.name`: `cnn1d` | `tcn` | `crnn`
- `train.device`: `cuda` 或 `cpu`
- `data`/`loader`/`train` 的其它超参

训练输出保存在 `runs/` 目录（包含 `best.pt` / `last.pt` / `history.json` / `config.yaml`）。

## 可视化与检查
随机可视化一个咳嗽样本：

```bash
python scripts/03_visualize_edgeai.py --mic out
```

检查数据加载与采样是否正常：

```bash
python scripts/06_check_dataloader.py
```

可视化训练历史曲线：

```bash
python scripts/08_visualize_history.py runs/your_run_name/history.json
```

## 目录结构
- `configs/` 训练与数据配置
- `data/` 原始/处理后的数据（默认忽略提交）
- `runs/` 训练产物（默认忽略提交）
- `scripts/` 数据与训练流水线脚本
- `src/coughcount/` 核心库代码

## 备注
如果在非仓库根目录运行脚本，可设置环境变量 `COUGHCOUNT_ROOT` 指向项目根目录。
