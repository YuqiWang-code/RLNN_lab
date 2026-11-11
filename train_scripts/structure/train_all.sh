#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

# Train all four ablation arms (two per GPU lane).
lane() {
  local gpu="$1"; shift
  for arm in "$@"; do
    CUDA_VISIBLE_DEVICES="$gpu" bash "$SCRIPTS/train.sh" "$arm" --gpu "$gpu" \
      > "$OUT/train_${arm}_ref${REFERENCE_SEED}.stdout" 2>&1
  done
}
lane 0 degree all &
first=$!
lane 1 gate gain &
second=$!
wait "$first"
wait "$second"
date -Is > "$OUT/train_completed.txt"
