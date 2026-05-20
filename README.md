# cough_count_e2e

基于 EdgeAI 咳嗽音频数据的端到端咳嗽次数估计。

## 项目结构

```
├── src/                    # 核心库代码
├── data/                   # 数据目录
├── experiments/            # 实验目录 (所有实验入口)
│   ├── 00_data_prep/       # 数据预处理 (运行一次)
│   ├── 01_baseline_tcn/    # 基线TCN模型
│   ├── 02_density_opt/     # Density优化实验
│   └── ...
└── configs/                # 共享配置文件
```

## 快速开始

当前本地可跑主线建议优先使用：

```text
experiments/03_loso_model_comparison
```

第5章正式重跑入口已整理在：

```text
experiments/03_loso_model_comparison/scripts/06_ch5_rerun_queue.py
experiments/03_loso_model_comparison/RERUN_0P7M_REMOTE_GUIDE.md
```

当前正式队列包含 `8` 个 0.7M 级模型结构横向对比和 `4` 个 0.7M 级 TCN/BiGRU 消融任务。四张 4090 单机运行时，启动 4 个 shard，每张卡串行跑 3 个 LOSO 任务。

### 1. 数据预处理 (首次运行)

进入 `experiments/00_data_prep/` 目录，按顺序运行脚本：

```bash
cd experiments/00_data_prep

# 下载EdgeAI数据集
python scripts/01_download.py

# 生成manifest.csv
python scripts/02_build_manifest.py

# 划分train/val/test受试者
python scripts/03_split_subjects.py

# 预计算STFT特征和density标签
python scripts/04_precompute_windows.py --mic both

# 可视化检查
python scripts/05_visualize.py --mic out

# 验证数据加载
python scripts/06_check_dataloader.py
```

### 2. 运行实验

每个实验都是独立的，进入对应的实验目录运行：

```bash
# 基线TCN模型
cd experiments/01_baseline_tcn
python scripts/01_train.py --config ../../configs/models/tcn.yaml

# Density优化实验
cd experiments/02_density_opt
python scripts/01_train.py --config ../../configs/edgeai_best_tcn_gru.yaml
```

### 3. 查看结果

每个实验的输出都在该实验目录的 `runs/` 子目录中：

- `runs/*/history.json` - 训练历史
- `runs/*/best.pt` - 最佳模型权重
- `runs/*/config.yaml` - 运行配置

## 实验

| 实验 | 说明 |
|------|------|
| `00_data_prep` | EdgeAI数据预处理 |
| `01_baseline_tcn` | 基线TCN模型实验 |
| `02_density_opt` | Density核函数和损失函数优化 |

## 环境与安装

```bash
# 使用 uv (推荐)
uv sync

# 或使用 pip
python -m pip install -e .
```

### 3090 运行建议

先确认 CUDA 可用：

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

最小烟测：

```powershell
cd experiments\03_loso_model_comparison
..\..\.venv\Scripts\python.exe scripts\03_loso.py --model-id M2 --device cuda --epochs 1 --max-folds 1 --batch-size 8 --num-workers 0
```

优先复跑第5章 0.7M 结构对比与消融：

```powershell
..\..\.venv\Scripts\python.exe scripts\06_ch5_rerun_queue.py audit
..\..\.venv\Scripts\python.exe scripts\06_ch5_rerun_queue.py run --shard-index 0 --num-shards 4 --device cuda:0 --dry-run
```

## 配置说明

### Density核函数参数

在 `scripts/04_precompute_windows.py` 中调整：

- `--kernel`: gaussian | skewed_gaussian | cosine
- `--sigma-sec`: 高斯核标准差 (默认 0.05)
- `--sigma-left-sec`: skewed_gaussian 左侧sigma (默认 0.04)
- `--sigma-right-sec`: skewed_gaussian 右侧sigma (默认 0.08)

### 训练参数

在对应的 `configs/*.yaml` 中调整：

- `model.name`: gru | tcn | tcn_gru | cnn1d | crnn
- `train.count_loss_weight`: count loss权重 (默认 0.2)
- `train.under_count_weight`: 假阳性惩罚权重 (默认 0.0)
- `data.pos_threshold`: 正样本阈值 (默认 0.01)
