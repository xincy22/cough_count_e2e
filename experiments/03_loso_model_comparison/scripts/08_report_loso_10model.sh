#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EXP_DIR"

PYTHON_BIN="${PYTHON_BIN:-../../.venv/bin/python}"

"$PYTHON_BIN" scripts/06_loso_10model_queue.py report
