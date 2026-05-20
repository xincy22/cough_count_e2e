#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EXP_DIR"

PYTHON_BIN="${PYTHON_BIN:-../../.venv/bin/python}"

echo "[1/3] Python: $PYTHON_BIN"
"$PYTHON_BIN" -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available(), 'gpus=', torch.cuda.device_count())"

echo "[2/3] Audit Chapter 5 rerun queue"
"$PYTHON_BIN" scripts/06_ch5_rerun_queue.py audit

echo "[3/3] Start four shard workers"
nohup "$PYTHON_BIN" scripts/06_ch5_rerun_queue.py run --shard-index 0 --num-shards 4 --device cuda:0 > shard0.out 2>&1 &
echo "started shard0 pid=$!"

nohup "$PYTHON_BIN" scripts/06_ch5_rerun_queue.py run --shard-index 1 --num-shards 4 --device cuda:1 > shard1.out 2>&1 &
echo "started shard1 pid=$!"

nohup "$PYTHON_BIN" scripts/06_ch5_rerun_queue.py run --shard-index 2 --num-shards 4 --device cuda:2 > shard2.out 2>&1 &
echo "started shard2 pid=$!"

nohup "$PYTHON_BIN" scripts/06_ch5_rerun_queue.py run --shard-index 3 --num-shards 4 --device cuda:3 > shard3.out 2>&1 &
echo "started shard3 pid=$!"

echo
echo "Monitor:"
echo "  nvidia-smi"
echo "  tail -f shard0.out"
echo "  tail -f runs/ch5_rerun_0p7m_logs/J00_cnn1d_0p7m.log"
echo
echo "After all shards finish:"
echo "  $PYTHON_BIN scripts/06_ch5_rerun_queue.py report"
