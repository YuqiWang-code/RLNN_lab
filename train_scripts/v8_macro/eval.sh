#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
arm="${1:?pass A2, A3 or A5}"
protocol="${2:-budget}"
mkdir -p "$OUT"
extra=()
if [[ "$protocol" == budget ]]; then extra=(--budget_s "${BUDGET:-1.0}"); else extra=(--steps "${STEPS:-50}"); fi
"$PYTHON" -u models/v8_eval.py --arm "$OUT/train_${arm}_ref${REFERENCE_SEED}" \
  --reference_seed "$REFERENCE_SEED" --save_dir "$BACKBONE" --result_dir "$OUT" \
  --split validation --test_samples 100 --seeds $SEEDS --protocol "$protocol" "${extra[@]}" 2>&1 \
  | tee "$OUT/eval_${arm}_${protocol}_ref${REFERENCE_SEED}.log"
