# Trial Workspace

本目录是独立实验空间，可直接按顺序执行脚本完成当前 trial 目标。

顺序执行：
1. `python scripts/01_download_edgeai.py`
2. `python scripts/02_build_manifest_edgeai.py`
3. `python scripts/03_visualize_edgeai.py`
4. `python scripts/04_split_edgeai_subjects.py`
5. `python scripts/05_precompute_edgeai_windows.py --mic both`
6. `python scripts/06_check_dataloader.py`
7. `python scripts/07_train_edgeai.py`
8. `python scripts/08_visualize_history.py runs/<run_name>/history.json`
9. `python scripts/09_test_edgeai.py runs/<run_name> --split test`

快捷训练：
- PowerShell: `./scripts/train.ps1 -RunName <run_name>`
- Bash: `./scripts/train.sh configs/edgeai.yaml <run_name>`

默认配置：`configs/edgeai.yaml`

配置整理规则：
- `configs/edgeai.yaml` 只放当前默认训练配置。
- `configs/finetune/` 放阶段微调配置。
- `configs/search/` 放搜索/扫描配置。
- `configs/archive/` 放历史配置（不默认使用）。

详细索引见 `configs/README.md`。
