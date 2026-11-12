#!/usr/bin/env bash
# KaMIS (ReduMIS) anytime curve at the deployed budgets.  Read-only measurement:
# nothing about the pipeline changes, this only answers "who wins at 0.5 s / 1.0 s".
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

KAMIS="${KAMIS:-$ROOT/tmp/kamis_build/redumis}"
BUDGETS="${BUDGETS:-0.1 0.5 1.0 4.0}"
RESULT="$OUT/positioning"

if [[ ! -x "$KAMIS" ]]; then
  echo "KaMIS binary not found or not executable: $KAMIS" >&2
  echo "set KAMIS=/path/to/redumis" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="" "$PYTHON" -u models/eval_kamis_anytime.py \
  --split validation --test_samples 100 \
  --kamis "$KAMIS" --budgets $BUDGETS --seeds 1017 1019 \
  --reference_seed "$REFERENCE_SEED" \
  --result_dir "$RESULT" 2>&1 | tee -a "$OUT/queue/kamis_anytime.log"
