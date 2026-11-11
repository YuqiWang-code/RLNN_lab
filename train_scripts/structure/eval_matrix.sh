#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

mkdir -p "$OUT"
run_fixed() {
  local gpu="$1"; shift
  for arm in "$@"; do
    CUDA_VISIBLE_DEVICES="$gpu" CUDA_GRAPH=1 bash "$SCRIPTS/eval.sh" fixed "$arm" --gpu "$gpu" \
      > "$OUT/fixed_${arm}_ref${REFERENCE_SEED}.stdout" 2>&1
  done
}

run_fixed 0 baseline degree all &
first=$!
run_fixed 1 gate gain &
second=$!
wait "$first"
wait "$second"
date -Is > "$OUT/fixed_completed.txt"

run_budget() {
  local gpu="$1" budget="$2" arm="$3"
  BUDGET="$budget" CUDA_VISIBLE_DEVICES="$gpu" CUDA_GRAPH=1 \
    bash "$SCRIPTS/eval.sh" budget "$arm" --gpu "$gpu" \
    > "$OUT/budget${budget//./}_${arm}_ref${REFERENCE_SEED}.stdout" 2>&1
}

for budget in 0.5 1.0; do
  run_budget 0 "$budget" baseline &
  first=$!
  run_budget 1 "$budget" all &
  second=$!
  wait "$first"
  wait "$second"
done
date -Is > "$OUT/completed.txt"
