#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/edgeai.yaml}"
RUN_NAME="${2:-}"
INIT_CKPT="${3:-}"
RESUME="${4:-}"

if [[ -z "$CONFIG" ]]; then
  echo "Usage: ./train.sh <config_relpath> [run_name] [init_ckpt] [resume_true_or_false]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"
export COUGHCOUNT_WORKSPACE="$WORKSPACE"

CMD=(python "${SCRIPT_DIR}/07_train_edgeai.py" --config "$WORKSPACE/$CONFIG")
if [[ -n "$RUN_NAME" ]]; then
  CMD+=(--run-dir "$WORKSPACE/runs/$RUN_NAME")
fi
if [[ -n "$INIT_CKPT" ]]; then
  CMD+=(--init-ckpt "$INIT_CKPT")
fi
if [[ "$RESUME" == "true" ]]; then
  CMD+=(--resume)
fi

"${CMD[@]}"
