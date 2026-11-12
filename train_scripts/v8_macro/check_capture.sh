#!/usr/bin/env bash
# Gate the captured search before any speed claim: identical trajectory, then timing.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

mkdir -p "$OUT/queue"
"$PYTHON" -u models/check_capture_budget.py \
  --reference_seed "$REFERENCE_SEED" --save_dir "$BACKBONE" --split validation \
  --graphs "${GRAPHS:-2}" --steps "${STEPS:-20}" --chunk "${CHUNK:-10}" --repeat 3 \
  --result_dir "$OUT/check_capture" 2>&1 | tee -a "$OUT/queue/check_capture.log"
