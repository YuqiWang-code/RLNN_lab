#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
mkdir -p "$OUT"
for split in train validation; do
  samples=300; [[ "$split" == validation ]] && samples=100
  "$PYTHON" -u models/v8_collect.py --reference_seed "$REFERENCE_SEED" --save_dir "$BACKBONE" \
    --result_dir "$OUT" --data_split "$split" --test_samples "$samples" --horizon "$HORIZON" \
    --seeds "${SEEDS%% *}" 2>&1 | tee "$OUT/collect_${split}_ref${REFERENCE_SEED}.log"
done
