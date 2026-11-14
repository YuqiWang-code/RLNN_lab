#!/usr/bin/env bash
# Train the flip policy with a pooled graph token (``--arch global``).
#
# The frozen backbone is a five layer GCN; on a 231 node RB graph a node sees a
# small part of the instance, and the plateau study showed the policy settles at
# a local optimum whose one-flip and two-swap neighbourhoods are both empty.
# This arm adds mean/max pooled context to every node and keeps everything else
# identical (same loss, same step budget, same data), so the only variable is
# global context.
#
#   EPOCHS=50 RUN_NAME=global_tp50 bash train_scripts/v8_macro/global_train.sh
set -uo pipefail
cd "${RLNN_ROOT:-/home/hzeng/project/ZH/RLNN_lab}" || exit 1
source train_scripts/v8_macro/common.sh

export CUDA_VISIBLE_DEVICES="${V8_GPU:-1}"
mkdir -p "$OUT/queue"

RUN_NAME="${RUN_NAME:-global_tp50}"
EPOCHS="${EPOCHS:-50}"
GRAPHS="${GRAPHS:-1000}"
NUM_TP="${NUM_TP:-50}"
LR="${LR:-1e-4}"
BATCH="${BATCH:-64}"
GRAPH_BATCH="${GRAPH_BATCH:-8}"
NUM_K="${NUM_K:-20}"
NUM_H="${NUM_H:-128}"
NUM_L="${NUM_L:-5}"
LAMBD="${LAMBD:-0.5}"

"$PYTHON" -u "$ROOT/tools/run_rlnn.py" train --version v2 --run "$RUN_NAME" --arch global \
  --frequency none --seed 0 --epochs "$EPOCHS" \
  --graph_batch_size "$GRAPH_BATCH" --batch_size "$BATCH" --lr "$LR" \
  --validation_seed 1017 \
  --train_path "$ROOT/models/data/MIS_tensors/RB_small_train" --train_samples "$GRAPHS" \
  --valid_path "$ROOT/models/data/MIS_tensors/RB_small_val" --valid_samples 100 \
  --test_path "$ROOT/models/data/author_rld4co/MIS/RB_small_test" \
  --num_workers "${WORKERS:-4}" \
  --num_k "$NUM_K" --num_t 100 --num_tp "$NUM_TP" --num_d 5 --num_l "$NUM_L" --num_h "$NUM_H" \
  --mixed_precision no --beta 1.02 --lambd "$LAMBD" \
  "$@" 2>&1 | tee "$OUT/queue/global_${RUN_NAME}.log"
