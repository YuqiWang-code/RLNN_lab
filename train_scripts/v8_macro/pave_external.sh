#!/usr/bin/env bash
# Is PAVE solver-agnostic?  Feed KaMIS solutions into the frozen post-processing.
#   KAMIS_LIMIT=0.1 PAVE_LIMIT=0.1 bash train_scripts/v8_macro/pave_external.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/hzeng/envs/rlnn/bin/python}"
OUT="${OUT_V8:-$ROOT/outputs/v8_macro}"
REFERENCE_SEED="${REFERENCE_SEED:-0}"
KAMIS="${KAMIS:-$ROOT/tmp/kamis_build/redumis}"
GRAPHS="${GRAPHS:-100}"
SAMPLES="${SAMPLES:-3}"
SEEDS_LIST="${SEEDS_LIST:-1017}"
SOURCE="${SOURCE:-kamis}"
KAMIS_LIMIT="${KAMIS_LIMIT:-0.1}"
PAVE_LIMIT="${PAVE_LIMIT:-0.1}"
RESULT_DIR="${RESULT_DIR:-$OUT}"
mkdir -p "$RESULT_DIR/queue"
export CUDA_VISIBLE_DEVICES=""
cd "$ROOT"
"$PYTHON" -u models/eval_pave_external.py \
  --kamis "$KAMIS" --source "$SOURCE" --split validation --test_samples "$SAMPLES" \
  --kamis_limit "$KAMIS_LIMIT" --pave_limit "$PAVE_LIMIT" \
  --seeds $SEEDS_LIST --reference_seed "$REFERENCE_SEED" \
  --result_dir "$RESULT_DIR" 2>&1 | tee "$RESULT_DIR/queue/pave_external_${SOURCE}.log"
