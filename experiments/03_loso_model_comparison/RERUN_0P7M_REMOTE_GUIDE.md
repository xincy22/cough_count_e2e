# Chapter 5 0.7M Rerun Remote Guide

This is the only entry guide for the new Chapter 5 rerun.

## Remote Setup

Run all commands from the project virtual environment.

Upload package from the Windows machine:

```powershell
scp -P <port> C:\Users\xincy\Documents\Projects\wst\cough_count_e2e_ch5_rerun_0p7m_code.zip <user>@<host>:/root/
scp -P <port> C:\Users\xincy\Documents\Projects\wst\edgeai_loso_precomputed_data_03.zip <user>@<host>:/root/
```

Unpack on the remote machine:

```bash
mkdir -p /root/cough_count_e2e
unzip -o /root/cough_count_e2e_ch5_rerun_0p7m_code.zip -d /root/cough_count_e2e

mkdir -p /root/cough_count_e2e/experiments/03_loso_model_comparison/data
unzip -o /root/edgeai_loso_precomputed_data_03.zip -d /root/cough_count_e2e/experiments/03_loso_model_comparison/data
```

```bash
cd /root/cough_count_e2e

# If the machine already has uv:
uv sync

# If uv is not installed, use the image Python only to create the project venv:
python -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e .

./.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

Expected data directory:

```text
/root/cough_count_e2e/experiments/03_loso_model_comparison/data/
```

If this directory is missing, copy it from the previous machine or run the data-preparation pipeline before starting LOSO.

## What Will Run

The rerun is intentionally self-contained:

- Structure comparison: 8 models, all around 0.7M parameters.
- Ablation: 4 models, all around 0.7M parameters.
- Total: 12 LOSO jobs.

The two duplicated concepts (`TCN` and `TCN+BiGRU`) are deliberately rerun in both experiment configs so the structure table and ablation table each have their own complete raw evidence.

## Machine Layout

For one machine with four RTX4090 cards:

```bash
cd /root/cough_count_e2e/experiments/03_loso_model_comparison
bash scripts/07_start_ch5_rerun_4gpu.sh
```

Equivalent manual commands:

```bash
cd /root/cough_count_e2e/experiments/03_loso_model_comparison
nohup ../../.venv/bin/python scripts/06_ch5_rerun_queue.py run --shard-index 0 --num-shards 4 --device cuda:0 > shard0.out 2>&1 &
nohup ../../.venv/bin/python scripts/06_ch5_rerun_queue.py run --shard-index 1 --num-shards 4 --device cuda:1 > shard1.out 2>&1 &
nohup ../../.venv/bin/python scripts/06_ch5_rerun_queue.py run --shard-index 2 --num-shards 4 --device cuda:2 > shard2.out 2>&1 &
nohup ../../.venv/bin/python scripts/06_ch5_rerun_queue.py run --shard-index 3 --num-shards 4 --device cuda:3 > shard3.out 2>&1 &
```

Each shard runs 3 models sequentially on one GPU.

Old two-machine layout, only if needed:

```bash
cd /root/cough_count_e2e/experiments/03_loso_model_comparison
nohup ../../.venv/bin/python scripts/06_ch5_rerun_queue.py run --shard-index 0 --num-shards 4 --device cuda:0 > shard0.out 2>&1 &
nohup ../../.venv/bin/python scripts/06_ch5_rerun_queue.py run --shard-index 1 --num-shards 4 --device cuda:1 > shard1.out 2>&1 &
```

```bash
cd /root/cough_count_e2e/experiments/03_loso_model_comparison
nohup ../../.venv/bin/python scripts/06_ch5_rerun_queue.py run --shard-index 2 --num-shards 4 --device cuda:0 > shard2.out 2>&1 &
nohup ../../.venv/bin/python scripts/06_ch5_rerun_queue.py run --shard-index 3 --num-shards 4 --device cuda:1 > shard3.out 2>&1 &
```

## Before Running

Audit the queue:

```bash
cd /root/cough_count_e2e/experiments/03_loso_model_comparison
../../.venv/bin/python scripts/06_ch5_rerun_queue.py audit
```

Dry-run a shard command:

```bash
../../.venv/bin/python scripts/06_ch5_rerun_queue.py run --shard-index 0 --num-shards 4 --device cuda:0 --dry-run
```

## During Running

Check processes:

```bash
nvidia-smi
ps -ef | grep 06_ch5_rerun_queue.py | grep -v grep
```

Check logs:

```bash
tail -f shard0.out
tail -f runs/ch5_rerun_0p7m_logs/J00_cnn1d_0p7m.log
```

## After All Shards Finish

Generate the final report:

```bash
cd /root/cough_count_e2e/experiments/03_loso_model_comparison
../../.venv/bin/python scripts/06_ch5_rerun_queue.py report
```

Output location:

```text
result/ch5_rerun_0p7m_<timestamp>/
```

Important files:

```text
all_summary.csv
structure_summary.csv
ablation_summary.csv
missing_jobs.csv
README.md
```

## Expected Time

With 4 GPUs:

```text
12 jobs / 4 GPUs = 3 jobs per GPU
1 job about 6 h
Expected wall time about 18-24 h
```
