# Experiment 03: Chapter 5 LOSO 10-Model Comparison

本实验用于论文第五章“咳嗽事件计数模型”部分。当前项目主线已经收敛为 **统一参数规模下的 10 个模型结构横评**，不再单独写 TCN/GRU 消融实验。

## 实验目标

在相同数据、相同 LOSO 划分、相同训练协议和约 0.7M 参数规模下，对比 10 类咳嗽计数模型：

```text
CNN1D / DSCNN / ResCNN / CRNN / BiCRNN / BiGRU / TCN / TCN-Attn / TCN+UniGRU / TCN+BiGRU
```

该实验回答的问题是：

> 在主体独立的 LOSO 验证下，哪类模型结构更适合端到端咳嗽事件计数？

## 关键文件

| 文件 | 作用 |
|---|---|
| `experiment.yaml` | 默认 10 模型 LOSO 配置 |
| `configs/structure_compare_v2_0p7m.yaml` | 正式远端队列使用的同一份 10 模型配置 |
| `configs/ch5_rerun_jobs_0p7m.yaml` | 4 GPU 队列计划，10 个 job |
| `scripts/03_loso.py` | 单模型或多模型 LOSO 训练与测试 |
| `scripts/06_ch5_rerun_queue.py` | 审计、分片运行、汇总 release 报告的主入口 |
| `scripts/07_start_ch5_rerun_4gpu.sh` | 4 张 GPU 一键启动脚本 |
| `scripts/08_report_ch5_rerun.sh` | 训练完成后的报告生成脚本 |

## 共同实验设置

| 项目 | 设置 |
|---|---|
| 数据集 | EdgeAI cough-counting dataset |
| 验证方式 | 15-fold LOSO，每折留出 1 名受试者测试 |
| 输入特征 | STFT log-magnitude，形状 `[B, F, T]` |
| STFT | `win=1024`, `hop=256`, 单麦克风频率维 `F=513` |
| 麦克风 | `mic=both`，表示 out/body 两类麦克风样本均参与训练，不是通道拼接 |
| 窗口 | `window_sec=8.0`, `hop_sec=4.0` |
| 密度标签 | `skewed_gaussian`, `sigma_left_sec=0.03`, `sigma_right_sec=0.10` |
| 训练轮数 | `500` |
| batch size | `24` |
| dataloader workers | `4` |
| 优化器 | Adam，`lr=1e-3`, `weight_decay=0` |
| 学习率调度 | cosine cycle，`lr_cycle_epochs=100`, `lr_eta_min=1e-8` |
| checkpoint 选择 | 每折按 `val_count_mae` 选择 best checkpoint |
| 主指标 | `test_count_mae` |
| 次指标 | `test_count_mae_pos`, `test_count_mae_neg` |

## 模型配置

| ID | 模型 | 参数量 | 结构摘要 |
|---|---|---:|---|
| S0 | CNN1D | 726,641 | 5 层 Conv1d，channels `[48,96,160,224,288]` |
| S1 | DSCNN | 699,974 | 5 个 depthwise-separable residual CNN block |
| S2 | ResCNN | 699,905 | 3 个普通 residual CNN block，channels `[64,192,256]` |
| S3 | CRNN | 700,001 | CNN `[96,128,128,144,176]` + 单向 GRU hidden 256 |
| S4 | BiCRNN | 700,001 | CNN `[64,144,176,224,224]` + 双向 GRU hidden 112/方向 |
| S5 | BiGRU | 703,969 | `1x1 Conv` projection 288 + 双向 GRU hidden 192/方向 |
| S6 | TCN | 699,521 | 4 层 residual dilated TCN，channels 160，dilation `1,2,4,8` |
| S7 | TCN-Attn | 692,609 | 5 层 TCN + 1 层单头 self-attention |
| S8 | TCN+UniGRU | 699,297 | 4 层 TCN channels 128 + 单向 GRU hidden 224 |
| S9 | TCN+BiGRU | 698,209 | 4 层 TCN channels 128 + 双向 GRU hidden 144/方向 |

## 从零复现

在仓库根目录创建并使用项目虚拟环境：

```bash
uv sync --locked
```

准备数据：

```bash
python experiments/00_data_prep/scripts/01_download.py
python experiments/00_data_prep/scripts/02_build_manifest.py
python experiments/00_data_prep/scripts/03_split_subjects.py
cd experiments/03_loso_model_comparison
../../.venv/bin/python scripts/01_precompute.py
```

快速 smoke test：

```bash
../../.venv/bin/python scripts/06_ch5_rerun_queue.py audit
../../.venv/bin/python scripts/03_loso.py --model-id S9 --device cuda --epochs 1 --max-folds 1 --batch-size 8 --num-workers 0
```

4 张 GPU 正式运行：

```bash
cd experiments/03_loso_model_comparison
bash scripts/07_start_ch5_rerun_4gpu.sh
```

训练完成后生成 release 报告：

```bash
../../.venv/bin/python scripts/06_ch5_rerun_queue.py report
```

输出目录：

```text
experiments/03_loso_model_comparison/result/ch5_10model_compare_0p7m_<timestamp>/
```

报告目录会包含：

| 文件 | 作用 |
|---|---|
| `RELEASE_REPORT.md` | release 用完整实验报告 |
| `model_comparison_summary.csv` | 按 Count MAE 排序的模型级结果 |
| `fold_results_all.csv` | 150 条 fold-level 原始测试指标 |
| `environment.json` | Python / PyTorch / CUDA / GPU / git revision |
| `ch5_rerun_jobs_0p7m.yaml` | 精确 job 队列 |
| `structure_compare_v2_0p7m.yaml` | 精确模型与训练配置 |

## 论文结论口径

当前结果应写作“10 个模型结构横评”，不是严格组件消融。严格消融需要从同一个完整模型中逐项去除组件，而本实验是将每个完整模型调到接近 0.7M 参数规模后进行公平横向比较。

推荐结论：

> 在统一约 0.7M 参数规模下，BiCRNN 取得最低 Count MAE；TCN-Attn、TCN+UniGRU、TCN+BiGRU 表现接近，属于第一梯队。CRNN 与 BiGRU 也明显优于纯 CNN 类模型。整体结果说明，咳嗽计数任务需要显式时序建模，单纯卷积结构不足以稳定完成事件计数。
