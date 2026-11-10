#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
for marker in encoders adapted; do
  while [[ ! -f "$OUT/queue/$marker.done.txt" ]]; do sleep 5; done
done
"$PYTHON" train_scripts/legacy/v3/validate_heads.py > "$OUT/queue/validation_final.log" 2>&1
bash train_scripts/legacy/v3/reserve.sh pave 0 & first=$!
bash train_scripts/legacy/v3/reserve.sh random 1 & second=$!
wait "$first"
wait "$second"
"$PYTHON" train_scripts/legacy/v3/select_reserve.py > "$OUT/queue/budget_selection.log"
run_gpu() {
  "$PYTHON" train_scripts/legacy/v3/test_matrix.py --stage fixed --gpu "$1"
  "$PYTHON" train_scripts/legacy/v3/test_matrix.py --stage budget --gpu "$1"
}
run_gpu 0 > "$OUT/queue/test_gpu0.log" 2>&1 & first=$!
run_gpu 1 > "$OUT/queue/test_gpu1.log" 2>&1 & second=$!
wait "$first"
wait "$second"
date -Is > "$OUT/queue/v3.done.txt"
