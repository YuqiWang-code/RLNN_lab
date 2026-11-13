#!/usr/bin/env bash
# Same-card KaMIS re-measurement on the 1000-graph author test split.
# Short clocks: 1000 graphs (the claim's 0.5-1 s regime).  Long clocks: 100 graphs
# (RLNN reports 98 s / 1.64 min, KLSA 35 s -- we need the same wall-clock points).
# KaMIS is single-threaded and the box has 96 cores, so both grids are sharded.
set -euo pipefail
cd "${RLNN_ROOT:-/home/hzeng/project/ZH/RLNN_lab}" || exit 1
source train_scripts/v8_macro/common.sh

KAMIS=${KAMIS:-$ROOT/tmp/kamis_build/redumis}
TEST=$ROOT/models/data/MIS_tensors/RB_small_author_test
SHORT_SHARDS=${SHORT_SHARDS:-6}
LONG_SHARDS=${LONG_SHARDS:-5}
mkdir -p "$OUT/queue"

for shard in $(seq 0 $((SHORT_SHARDS - 1))); do
  nohup env CUDA_VISIBLE_DEVICES="" "$PYTHON" -u models/eval_kamis_anytime.py \
    --kamis "$KAMIS" --split test --test_path "$TEST" --test_samples 1000 \
    --budgets 0.5 1.0 --seeds 1017 \
    --shards "$SHORT_SHARDS" --shard "$shard" \
    --reference_dir "$OUT/kamis_author/none" --result_dir "$OUT/kamis_author" \
    > "$OUT/queue/kamis_author_short_shard${shard}.log" 2>&1 &
  echo "short shard $shard pid $!"
done

for shard in $(seq 0 $((LONG_SHARDS - 1))); do
  nohup env CUDA_VISIBLE_DEVICES="" "$PYTHON" -u models/eval_kamis_anytime.py \
    --kamis "$KAMIS" --split test --test_path "$TEST" --test_samples 100 \
    --budgets 5.0 35.0 98.0 --seeds 1017 \
    --shards "$LONG_SHARDS" --shard "$shard" \
    --reference_dir "$OUT/kamis_author_long/none" --result_dir "$OUT/kamis_author_long" \
    > "$OUT/queue/kamis_author_long_shard${shard}.log" 2>&1 &
  echo "long shard $shard pid $!"
done
