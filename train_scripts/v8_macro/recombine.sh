#!/usr/bin/env bash
# Can the exact recombination of the K particle solutions close the 0.037 gap to
# KLSA?  Three arms at the same clock: ``captured`` (current best), ``recombine``
# (search stops ``RESERVE`` seconds early, then decode + Koenig recombine + PAVE)
# and ``oracle`` (recombine after the clock, bounds what the operator could buy).
#
# Split across SHARDS processes for the same reason as author_test.sh; each shard
# writes ``recombine_ref<r>_shard<i>.csv`` for ``models/merge_v8_shards.py``.
set -euo pipefail
cd "${RLNN_ROOT:-/home/hzeng/project/ZH/RLNN_lab}" || exit 1
source train_scripts/v8_macro/common.sh
export CUDA_VISIBLE_DEVICES="${V8_GPU:-1}"

mkdir -p "$OUT/queue" "$OUT/recombine"
SHARDS=${SHARDS:-4}
for shard in $(seq 0 $((SHARDS - 1))); do
  nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON" -u models/eval_v8_recombine.py \
    --reference_seed "$REFERENCE_SEED" --save_dir "$BACKBONE" --split test \
    --test_path "$ROOT/models/data/MIS_tensors/RB_small_author_test" \
    --test_samples "${GRAPHS:-1000}" --budgets ${BUDGETS:-0.5 1.0} \
    --arms ${ARMS:-captured recombine oracle} \
    --reserve "${RESERVE:-0.05}" --decode_chunk "${DECODE_CHUNK:-4}" \
    --seeds ${SEEDS_LIST:-1017 1019 1021} \
    --shards "$SHARDS" --shard "$shard" \
    --result_dir "${RESULT_DIR:-$OUT/recombine}" \
    > "$OUT/queue/recombine_shard${shard}.log" 2>&1 &
  echo "shard $shard pid $!"
done
