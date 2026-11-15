#!/usr/bin/env bash
set -uo pipefail
cd "${RLNN_ROOT:-/home/hzeng/project/ZH/RLNN_lab}" || exit 1
source train_scripts/v8_macro/common.sh
export CUDA_VISIBLE_DEVICES="${V8_GPU:-1}"
mkdir -p "$OUT/queue"
"$PYTHON" -u models/probe_lns_headroom.py \
  --reference_seed "${REFERENCE_SEED:-0}" --save_dir "$BACKBONE" \
  --split test \
  --test_path "$ROOT/models/data/MIS_tensors/RB_small_author_test" \
  --test_samples "${GRAPHS:-20}" \
  --budgets ${BUDGETS:-1.0} \
  --radii ${RADII:-1 2} \
  --balls "${BALLS:-120}" \
  --seeds "${SEEDS_LIST:-1017}" \
  --result_dir "${RESULT_DIR:-$OUT/lns_headroom}" \
  --shards 1 --shard 0 \
  2>&1 | tee "$OUT/queue/lns_headroom.log"
