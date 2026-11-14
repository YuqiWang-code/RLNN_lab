#!/usr/bin/env bash
# Interleave the PAVE exchange move with the captured search loop.
set -euo pipefail
cd "${RLNN_ROOT:-/home/hzeng/project/ZH/RLNN_lab}" || exit 1
source train_scripts/v8_macro/common.sh

export CUDA_VISIBLE_DEVICES="${V8_GPU:-1}"
PYTHON="${PYTHON:-/home/hzeng/envs/rlnn/bin/python}"
BACKBONE="$OUT/../v2/none_seed0/checkpoints"
SHARDS="${SHARDS:-1}"
BUDGETS="${BUDGETS:-0.2 1.0}"
ARMS="${ARMS:-search inject refill}"
CYCLE_STEPS="${CYCLE_STEPS:-100}"
PAVE_SHARE="${PAVE_SHARE:-0.02}"
GRAPHS="${GRAPHS:-120}"
SEEDS_LIST="${SEEDS_LIST:-1017}"
RESULT_DIR="${RESULT_DIR:-$OUT/interleave}"
mkdir -p "$OUT/queue" "$RESULT_DIR"

for ((i = 0; i < SHARDS; i++)); do
  nohup "$PYTHON" models/eval_v8_interleave.py \
    --split test --test_path "$ROOT/models/data/MIS_tensors/RB_small_author_test" \
    --test_samples "$GRAPHS" --budgets $BUDGETS --arms $ARMS \
    --decoders cached --cycle_steps "$CYCLE_STEPS" --pave_share "$PAVE_SHARE" \
    --seeds $SEEDS_LIST --repeats 1 --chunk 10 --decode_chunk 4 --decode_every 2 \
    --save_dir "$BACKBONE" --reference_seed 0 \
    --result_dir "$RESULT_DIR" --shards "$SHARDS" --shard "$i" \
    > "$OUT/queue/interleave_shard${i}.log" 2>&1 &
  echo "shard $i -> pid $!"
done
wait
