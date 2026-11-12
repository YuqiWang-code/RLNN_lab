#!/usr/bin/env bash
# Equal-budget comparison of the eager step against the captured block replay.
# ARMS selects which engines run; BUDGETS is a space-separated list of seconds.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

mkdir -p "$OUT/queue"
"$PYTHON" -u models/eval_v8_capture_budget.py \
  --reference_seed "$REFERENCE_SEED" --save_dir "$BACKBONE" --split validation \
  --test_samples "${GRAPHS:-100}" --budgets ${BUDGETS:-0.5 1.0} \
  --arms ${ARMS:-eager captured} \
  --seeds ${SEEDS_LIST:-1017} --repeats "${REPEATS:-2}" \
  --result_dir "${RESULT_DIR:-$OUT/positioning}" 2>&1 | tee -a "$OUT/queue/capture_budget.log"
