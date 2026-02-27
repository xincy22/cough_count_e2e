# cough_count_e2e

当前项目已切换为 `trials/` 驱动的实验结构。  
根目录不再放 `configs/`、`scripts/`、`runs/`、`logs/`。

## 目录
- `src/` 核心代码
- `trials/tcn_tuning/`
- `trials/tcn_gru_tuning/`
- `trials/crnn_tuning/`

每个 trial 内都包含：
- `configs/`
- `scripts/`
- `runs/`
- `logs/`

## 使用方式（按顺序）
进入任一 trial（例如 `tcn_gru_tuning`）后，按顺序执行：

```bash
python scripts/01_download_edgeai.py
python scripts/02_build_manifest_edgeai.py
python scripts/03_visualize_edgeai.py
python scripts/04_split_edgeai_subjects.py
python scripts/05_precompute_edgeai_windows.py --mic both
python scripts/06_check_dataloader.py
python scripts/07_train_edgeai.py
python scripts/08_visualize_history.py runs/<run_name>/history.json
python scripts/09_test_edgeai.py runs/<run_name> --split test
```

也可用便捷脚本：
- PowerShell: `./scripts/train.ps1 -RunName <run_name>`
- Bash: `./scripts/train.sh configs/<config>.yaml <run_name>`

## 说明
- trial 脚本会自动把 workspace 绑定到当前 trial 目录（`COUGHCOUNT_WORKSPACE`）。
- 数据仍共享根目录下的 `data/`（由核心路径模块管理）。
