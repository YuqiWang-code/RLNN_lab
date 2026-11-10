#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec bash "$ROOT/train_scripts/legacy/v2/eval.sh" --split validation --test_samples 100 \
  --save_dir "$(base_checkpoint 0)" --label RLNN --post none rule heuristic random rule23 \
  --opportunities --result_dir "$OUT/diagnostics_exchange" "$@"
