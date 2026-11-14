#!/usr/bin/env bash
# Equal-per-graph-clock curve at the clocks the paper's baselines actually run at.
# RLNN: 98 s / 500 graphs = 0.196 s/graph; RLSA: 35 s / 500 = 0.07 s/graph.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

mkdir -p "$OUT/queue"
"$PYTHON" -u models/eval_v8_capture_budget.py \
  --reference_seed "$REFERENCE_SEED" --save_dir "$BACKBONE" --split test \
  --test_path "$ROOT/models/data/MIS_tensors/RB_small_author_test" \
  --test_samples "${GRAPHS:-1000}" --budgets ${BUDGETS:-0.05 0.1 0.2} \
  --arms ${ARMS:-captured} --decoders ${DECODERS:-cached} \
  --decode_chunk "${DECODE_CHUNK:-4}" \
  --seeds ${SEEDS_LIST:-1017 1019 1021} --repeats "${REPEATS:-1}" \
  --shards "${SHARDS:-6}" --shard "${SHARD:-0}" \
  --result_dir "${RESULT_DIR:-$OUT/author_short}" \
  2>&1 | tee -a "$OUT/queue/author_short_shard${SHARD:-0}.log"
