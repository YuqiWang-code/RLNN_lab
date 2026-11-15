#!/usr/bin/env bash
# Temperature gate for the annealed-policy line: sweep T on the amortised engine.
#
#   ANNEAL=1 TEMPERATURES="2 4" bash train_scripts/v8_macro/temperature.sh
#
# One process on purpose: four concurrent shards serialise the kernel launches
# and the step counts stop being comparable (see RLNN_v8_Amortised_Capture.md).
set -uo pipefail
cd "${RLNN_ROOT:-/home/hzeng/project/ZH/RLNN_lab}" || exit 1
source train_scripts/v8_macro/common.sh

export CUDA_VISIBLE_DEVICES="${V8_GPU:-1}"
mkdir -p "$OUT/queue"
EXTRA=()
if [ "${ANNEAL:-0}" = "1" ]; then
  EXTRA+=(--anneal)
fi

"$PYTHON" -u models/eval_v8_padded.py \
  --reference_seed "${REFERENCE_SEED:-0}" --save_dir "$BACKBONE" \
  --split test \
  --test_path "$ROOT/models/data/MIS_tensors/RB_small_author_test" \
  --test_samples "${GRAPHS:-100}" \
  --budgets ${BUDGETS:-0.2 1.0} \
  --decoders ${DECODERS:-cached} \
  --temperatures ${TEMPERATURES:-1 2 4 8} \
  --seeds ${SEEDS_LIST:-1017} \
  --repeats "${REPEATS:-1}" \
  --decode_every "${DECODE_EVERY:-1}" \
  "${EXTRA[@]}" \
  --result_dir "${RESULT_DIR:-$OUT/temperature}" \
  --shards 1 --shard 0 \
  2>&1 | tee "$OUT/queue/temperature.log"
