#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
export PYTORCH_ALLOC_CONF=backend:native
seed="${SEED:-0}"
run="${RUN_NAME:-pave_seed${seed}}"
exec "$PYTHON" -u "$ROOT/models/train_exchange_value.py" --train_seed "$seed" \
  --save_dir "$(base_checkpoint "$seed")" --result_dir "$OUT/$run" "$@"
