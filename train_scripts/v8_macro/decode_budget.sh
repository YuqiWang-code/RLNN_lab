#!/usr/bin/env bash
# Does the captured A.3 decode survive the swap?  Four cells, paired per graph:
# eager/eager (the reference), captured/eager (the old result), captured/captured
# (the new decoder) and eager/captured (the decoder on its own).
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

mkdir -p "$OUT/queue"
"$PYTHON" -u models/eval_v8_capture_budget.py \
  --reference_seed "$REFERENCE_SEED" --save_dir "$BACKBONE" --split validation \
  --test_samples "${GRAPHS:-100}" --budgets ${BUDGETS:-0.5 1.0} \
  --arms ${ARMS:-eager captured} --decoders ${DECODERS:-eager captured cached} \
  --decode_chunk "${DECODE_CHUNK:-4}" \
  --seeds ${SEEDS_LIST:-1017} --repeats "${REPEATS:-2}" \
  --result_dir "${RESULT_DIR:-$OUT/positioning_decode}" \
  2>&1 | tee -a "$OUT/queue/decode_budget.log"
