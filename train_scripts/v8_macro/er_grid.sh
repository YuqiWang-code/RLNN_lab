#!/usr/bin/env bash
# ER-[700-800] at the density that reproduces the pooled-table scale:
# KaMIS 1 s = 44.00, RLNN 43.34, KLSA 44.10, KaMIS 52 min = 44.87 (paper).
# n~750 graphs are ~4x the training size, so the grid is sharded.
set -euo pipefail
cd "${RLNN_ROOT:-/home/hzeng/project/ZH/RLNN_lab}" || exit 1
source train_scripts/v8_macro/common.sh
# GPU0 is shared with a long-running job from another user; the v8 grids
# default to the idle card.  Override with V8_GPU=0 when GPU1 is busy.
export CUDA_VISIBLE_DEVICES="${V8_GPU:-1}"

GRAPHS=${GRAPHS:-128}
SEEDS_LIST=${SEEDS_LIST:-1017 1019 1021}
BUDGETS=${BUDGETS:-0.5 1.0}
SHARDS=${SHARDS:-2}

if [ ! -d "$ROOT/models/data/MIS_tensors/ER_700_800" ]; then
  "$PYTHON" models/data/er_generator.py --min-n 700 --max-n 800 --p 0.15 \
      --num-graphs "$GRAPHS" --processes 8 --seed 0 \
      --output-dir "$ROOT/models/data/raw_regime/ER_700_800"
  "$PYTHON" models/data/prepare_mis_tensors.py --source "$ROOT/models/data/raw_regime/ER_700_800" \
      --destination "$ROOT/models/data/MIS_tensors/ER_700_800"
fi

mkdir -p "$OUT/queue"
for shard in $(seq 0 $((SHARDS - 1))); do
  nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON" -u models/eval_v8_capture_budget.py \
    --reference_seed "$REFERENCE_SEED" --save_dir "$BACKBONE" \
    --split validation --test_path "$ROOT/models/data/MIS_tensors/ER_700_800" \
    --test_samples "$GRAPHS" --budgets $BUDGETS \
    --arms eager captured --decoders eager cached \
    --seeds $SEEDS_LIST --repeats 1 \
    --shards "$SHARDS" --shard "$shard" \
    --result_dir "$OUT/er700_800" \
    > "$OUT/queue/er700_800_shard${shard}.log" 2>&1 &
  echo "er shard $shard pid $!"
done
