#!/usr/bin/env bash
# Gate the captured A.3 decoder: identical bitsets first, then the chunk sweep.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

mkdir -p "$OUT/queue"
"$PYTHON" -u models/check_decode_capture.py \
  --reference_seed "$REFERENCE_SEED" --save_dir "$BACKBONE" --split validation \
  --graphs 4 --repeat 5 --result_dir "$OUT/decode_cost" \
  2>&1 | tee -a "$OUT/queue/check_decode.log"
