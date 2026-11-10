#!/usr/bin/env bash
set -euo pipefail
arm="${1:?pass baseline or a v5 arm}"
shift
export PRECISION=fp16
export RUN_NAME="${RUN_NAME:-eval_budget_${BUDGET:-0.5}_${arm}_seed${SEED:-0}_${POST:-pave}}"
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/eval.sh" "$arm" \
  --post "${POST:-pave}" --time-budget-s "${BUDGET:-0.5}" --post-reserve-s 0.05 "$@"
