#!/usr/bin/env bash
# Where do the decode milliseconds go? Pure measurement, no training.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

mkdir -p "$OUT/queue"
"$PYTHON" -u models/probe_decode_cost.py \
  --reference_seed "$REFERENCE_SEED" --save_dir "$BACKBONE" --split validation \
  --graphs 3 --repeat 5 --result_dir "$OUT/decode_cost" \
  2>&1 | tee -a "$OUT/queue/decode_cost.log"
