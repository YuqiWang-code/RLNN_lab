#!/usr/bin/env bash
set -euo pipefail
arm="${1:?pass baseline or a v7 context arm}"
shift
budget="${BUDGET:?set BUDGET to 0.5 or 1.0}"
RUN_NAME="${RUN_NAME:-budget${budget/./}_${arm}_ref${REF_SEED:-0}}" exec bash "$(dirname -- "${BASH_SOURCE[0]}")/eval.sh" \
  "$arm" --post pave --time-budget-s "$budget" --extended-search "$@"
