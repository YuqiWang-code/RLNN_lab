#!/usr/bin/env bash
set -euo pipefail
arm="${1:?pass none or a v4 frequency arm}"
shift
export RUN_NAME="${RUN_NAME:-budget_${arm}_seed${SEED:-0}_${SPLIT:-test}_${POST:-pave}}"
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/eval.sh" "$arm" \
  --post "${POST:-pave}" --time-budget-s 0.5 --post-reserve-s 0.05 "$@"
