#!/usr/bin/env bash
# v9: train the flip policy with the density-compensated scale-space operator.
#
# The frozen backbone is a five layer GCN whose aggregation is a degree
# normalised *mean* (``models/models.py:46``).  On the author RB graphs the mean
# degree is ~80 (``capacity = 105`` at ``nodes=264``), so each further hop shrinks
# the node-to-node contrast by ~1/9, ~1/80, ~1/700; beyond one hop the network is
# averaging noise, which is why every diversity patch measured zero.  ``--arch
# scale`` (``models/mis_v9_scale.py``) reads the hops as explicit counts and
# rescales each one by its expected size, so three hops stay readable.
#
# Everything else is held fixed so the only variable is the operator: same loss,
# same 50 step rollouts, same data, same seed, same learning rate.  ``ARCH=gcn``
# with an identical command line is the control arm.
#
#   ARCH=scale RUN_NAME=v9_scale_tp50 bash train_scripts/v9/scale_train.sh
#   ARCH=gcn   RUN_NAME=v9_gcn_tp50   bash train_scripts/v9/scale_train.sh
set -uo pipefail
cd "${RLNN_ROOT:-/home/hzeng/project/ZH/RLNN_lab}" || exit 1
source train_scripts/v8_macro/common.sh

export CUDA_VISIBLE_DEVICES="${V8_GPU:-1}"
mkdir -p "$OUT/queue"

ARCH="${ARCH:-scale}"
RUN_NAME="${RUN_NAME:-v9_${ARCH}_tp50}"
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
SS_HOPS="${SS_HOPS:-3}"
SS_TAU="${SS_TAU:-2.0}"

"$PYTHON" -u "$ROOT/tools/run_rlnn.py" train --version v2 --run "$RUN_NAME" --arch "$ARCH" \
  --ss_hops "$SS_HOPS" --ss_tau "$SS_TAU" \
  --frequency none --seed 0 --epochs "$EPOCHS" \
  --graph_batch_size "$GRAPH_BATCH" --batch_size "$BATCH" --lr "$LR" \
  --validation_seed 1017 \
  --train_path "$ROOT/models/data/MIS_tensors/RB_small_train" --train_samples "$GRAPHS" \
  --valid_path "$ROOT/models/data/MIS_tensors/RB_small_val" --valid_samples 100 \
  --test_path "$ROOT/models/data/author_rld4co/MIS/RB_small_test" \
  --num_workers "${WORKERS:-4}" \
  --num_k "$NUM_K" --num_t 100 --num_tp "$NUM_TP" --num_d 5 --num_l "$NUM_L" --num_h "$NUM_H" \
  --mixed_precision no --beta 1.02 --lambd "$LAMBD" \
  "$@" 2>&1 | tee "$OUT/queue/v9_${ARCH}_${RUN_NAME}.log"
