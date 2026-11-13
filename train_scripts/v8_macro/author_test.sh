#!/usr/bin/env bash
# The headline grid: the 1000 author test graphs, three search seeds, the eager
# engine and the captured engine at the same clock. This is the table that has
# to hold up against KLSA 19.97 @35 s and RLNN 19.52 @98 s.
#
# The GPU loop is not saturated by one process (each one needs ~1.5 GiB and the
# python side is serial), so the graph list is split across SHARDS processes.
# Every shard writes ``capture_budget_ref<r>_shard<i>.csv``; merge with
# ``python models/merge_v8_shards.py``.
set -euo pipefail
cd "${RLNN_ROOT:-/home/hzeng/project/ZH/RLNN_lab}" || exit 1
source train_scripts/v8_macro/common.sh
# GPU0 is shared with a long-running job from another user; the v8 grids
# default to the idle card.  Override with V8_GPU=0 when GPU1 is busy.
export CUDA_VISIBLE_DEVICES="${V8_GPU:-1}"

mkdir -p "$OUT/queue"
SHARDS=${SHARDS:-4}
for shard in $(seq 0 $((SHARDS - 1))); do
  nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON" -u models/eval_v8_capture_budget.py \
    --reference_seed "$REFERENCE_SEED" --save_dir "$BACKBONE" --split test \
    --test_path "$ROOT/models/data/MIS_tensors/RB_small_author_test" \
    --test_samples "${GRAPHS:-1000}" --budgets ${BUDGETS:-0.5 1.0} \
    --arms ${ARMS:-eager captured} --decoders ${DECODERS:-eager cached} \
    --seeds ${SEEDS_LIST:-1017 1019 1021} --repeats "${REPEATS:-1}" \
    --shards "$SHARDS" --shard "$shard" \
    --result_dir "${RESULT_DIR:-$OUT/author_test}" \
    > "$OUT/queue/author_test_shard${shard}.log" 2>&1 &
  echo "shard $shard pid $!"
done
