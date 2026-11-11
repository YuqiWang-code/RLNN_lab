#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

# Deadline arms: same total budget, recombination takes its slice out of the search.
v7_args "$@"
mkdir -p "$OUT"
log="$OUT/budget_ref${REFERENCE_SEED}.log"
BUDGETS="${BUDGETS:-0.5 1.0}"
args=(--reference_seed "$REFERENCE_SEED" --save_dir "$(backbone_dir)" --result_dir "$OUT"
      --split validation --test_samples "$SAMPLES" --seeds $SEEDS --budgets $BUDGETS
      --phase recombine)

"$PYTHON" -u models/eval_v7_budget.py "${args[@]}" "${EXTRA[@]}" 2>&1 | tee "$log"
"$PYTHON" "$SCRIPTS/summarize_budget.py" --root "$OUT" --refs "$REFERENCE_SEED"
