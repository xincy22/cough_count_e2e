# Chapter 5 10-Model LOSO Comparison Release Report

生成日期：2026-05-21

## 1. 实验目的

本实验用于支撑第五章咳嗽事件计数模型部分。实验不再写作 TCN/GRU 消融，而是作为统一参数规模下的 10 个完整模型结构横评。核心问题是：在主体独立的 LOSO 验证下，哪类模型结构更适合端到端咳嗽事件计数。

## 2. 可复现实验设置

| 项目 | 设置 |
|---|---|
| 数据集 | EdgeAI cough-counting dataset |
| 验证方式 | 15-fold leave-one-subject-out，15 名受试者每人作为一次测试集 |
| 输入特征 | STFT log-magnitude，模型输入 `[B, F, T]` |
| STFT | `win=1024`, `hop=256`, 单麦克风频率维 `F=513` |
| 麦克风 | `mic=both`，out/body 两类麦克风样本均参与训练，不是通道拼接 |
| 窗口 | `window_sec=8.0`, `hop_sec=4.0` |
| 标签 | 帧级咳嗽密度序列，`skewed_gaussian`, `sigma_left_sec=0.03`, `sigma_right_sec=0.10` |
| 输出 | 帧级密度预测，经时间维积分得到窗口内预测咳嗽次数 |
| 训练 | 500 epochs, batch size 24, num_workers 4, Adam lr 1e-3, weight_decay 0 |
| 模型选择 | 每折按 `val_count_mae` 保存 best checkpoint |
| 主指标 | `test_count_mae` |
| 次指标 | `test_count_mae_pos`, `test_count_mae_neg` |
| 参数规模 | 全部模型约 0.7M trainable parameters |
| 远端硬件 | 4 x RTX 4090 24 GB，租用镜像显示 PyTorch 2.7.0 / Ubuntu 24.04 |

## 3. 模型结构

| ID | 模型 | 参数量 | 主体结构 | 关键配置 |
|---|---|---:|---|---|
| S0 | CNN1D | 726,641 | Conv1d-BN-ReLU-Dropout x5 | `channels=[48,96,160,224,288]; kernel=5; head=1x1 Conv` |
| S1 | DSCNN | 699,974 | Depthwise-separable residual CNN | `channels=[128,160,224,224,320]; kernel=3; 5 DS residual blocks` |
| S2 | ResCNN | 699,905 | Residual CNN | `channels=[64,192,256]; kernel=3; 3 residual blocks` |
| S3 | CRNN | 700,001 | CNN + unidirectional GRU | `cnn_channels=[96,128,128,144,176]; kernel=3; gru_hidden=256` |
| S4 | BiCRNN | 700,001 | CNN + bidirectional GRU | `cnn_channels=[64,144,176,224,224]; kernel=3; gru_hidden=112 per direction` |
| S5 | BiGRU | 703,969 | Pointwise projection + bidirectional GRU | `proj_channels=288; gru_hidden=192 per direction` |
| S6 | TCN | 699,521 | Residual dilated TCN | `channels=160; layers=4; kernel=3; dilations=[1,2,4,8]` |
| S7 | TCN-Attn | 692,609 | TCN + self-attention | `channels=128; tcn_layers=5; dilations=[1,2,4,8,16]; attn_heads=1; attn_layers=1` |
| S8 | TCN+UniGRU | 699,297 | TCN + unidirectional GRU | `tcn_channels=128; tcn_layers=4; dilations=[1,2,4,8]; gru_hidden=224` |
| S9 | TCN+BiGRU | 698,209 | TCN + bidirectional GRU | `tcn_channels=128; tcn_layers=4; dilations=[1,2,4,8]; gru_hidden=144 per direction` |

## 4. 模型对比结果

| 排名 | 模型 | 参数量 | LOSO folds | Count MAE | Pos MAE | Neg MAE |
|---:|---|---:|---:|---:|---:|---:|
| 1 | BiCRNN | 700,001 | 15 | 0.4735 ± 0.4684 | 0.9828 | 0.3185 |
| 2 | TCN-Attn | 692,609 | 15 | 0.5205 ± 0.4859 | 1.2564 | 0.3000 |
| 3 | TCN+UniGRU | 699,297 | 15 | 0.5208 ± 0.4820 | 1.1965 | 0.3211 |
| 4 | TCN+BiGRU | 698,209 | 15 | 0.5281 ± 0.4333 | 1.2991 | 0.2980 |
| 5 | CRNN | 700,001 | 15 | 0.5574 ± 0.4371 | 1.3512 | 0.3188 |
| 6 | BiGRU | 703,969 | 15 | 0.5747 ± 0.3900 | 1.4666 | 0.3007 |
| 7 | TCN | 699,521 | 15 | 0.7887 ± 0.4029 | 1.9368 | 0.4525 |
| 8 | DSCNN | 699,974 | 15 | 0.9398 ± 0.5342 | 1.8814 | 0.6622 |
| 9 | CNN1D | 726,641 | 15 | 1.0931 ± 0.4528 | 2.0426 | 0.8176 |
| 10 | ResCNN | 699,905 | 15 | 1.1328 ± 0.5642 | 2.0101 | 0.8751 |

## 5. 结果结论

BiCRNN 在本轮统一 0.7M 参数规模实验中取得最低 Count MAE，为 0.4735。TCN-Attn、TCN+UniGRU 与 TCN+BiGRU 的结果非常接近，Count MAE 分别为 0.5205、0.5208 和 0.5281，属于第一梯队。CRNN 与 BiGRU 也明显优于纯 CNN 类模型。

纯 CNN 类模型 CNN1D、DSCNN 和 ResCNN 的 Count MAE 较高，说明咳嗽计数不是简单的局部声学片段回归问题。引入循环结构、注意力机制或混合时序建模后，模型能更好地利用窗口内咳嗽事件的持续性、间隔关系和局部聚集特征。

因此，第五章建议将本实验表述为完整模型结构横评，而不是严格组件消融。严格组件消融需要从同一个完整模型中逐项去除模块；本实验的设计目标是控制总参数规模后比较不同完整结构。

## 6. Release 文件说明

| 文件 | 作用 |
|---|---|
| `model_comparison_summary.csv` | 10 个模型的主结果表，按 Count MAE 排名 |
| `fold_results_all.csv` | 150 条 fold-level 原始测试指标 |
| `model_architecture_summary.csv` | 每个模型的参数量和结构配置 |
| `reproducibility_manifest.json` | 数据、训练、环境、配置路径的机器可读说明 |
| `structure_compare_v2_0p7m.yaml` | 完整模型和训练配置 |
| `ch5_rerun_jobs_0p7m.yaml` | 4 GPU 队列计划 |
| `source_manifest.json` | 原始远端 fold JSON 来源路径 |

## 7. 复现命令

```bash
uv sync --locked
cd experiments/03_loso_model_comparison
../../.venv/bin/python scripts/01_precompute.py
../../.venv/bin/python scripts/06_ch5_rerun_queue.py audit
bash scripts/07_start_ch5_rerun_4gpu.sh
../../.venv/bin/python scripts/06_ch5_rerun_queue.py report
```
