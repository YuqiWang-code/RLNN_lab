#!/usr/bin/env bash
# One capture for the whole dataset: the amortised-prelude arm on the author test set.
set -uo pipefail
cd "${RLNN_ROOT:-/home/hzeng/project/ZH/RLNN_lab}" || exit 1
source train_scripts/v8_macro/common.sh

SHARDS="${SHARDS:-4}"
export CUDA_VISIBLE_DEVICES="${V8_GPU:-1}"
mkdir -p "$OUT/queue"

for i in $(seq 0 $((SHARDS - 1))); do
  nohup "$PYTHON" -u models/eval_v8_padded.py \
    --reference_seed "${REFERENCE_SEED:-0}" --save_dir "$BACKBONE" \
    --split test \
    --test_path "$ROOT/models/data/MIS_tensors/RB_small_author_test" \
    --test_samples "${GRAPHS:-1000}" \
    --budgets ${BUDGETS:-0.05 0.1 0.2 1.0} \
    --decoders ${DECODERS:-captured cached} \
    --seeds ${SEEDS_LIST:-1017 1019 1021} \
    --repeats "${REPEATS:-1}" \
    --result_dir "${RESULT_DIR:-$OUT/padded_author}" \
    --shards "$SHARDS" --shard "$i" \
    > "$OUT/queue/padded_shard${i}.log" 2>&1 &
done
wait
