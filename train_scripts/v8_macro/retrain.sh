#!/usr/bin/env bash
# Retrain (or fine-tune) the flip policy so that it converges inside the budget
# we actually deploy: 100-400 steps instead of 3000.
#
# The frozen v2 backbone was fine-tuned with ``--num_tp 50``: every training state
# came from the first 50 steps of the policy's own rollout.  Deployment runs to
# thousands of steps, i.e. almost entirely off that distribution.  This launcher
# sweeps the rollout length (``NUM_TP``) and the training budget.
#
#   NUM_TP=300 EPOCHS=5 RUN_NAME=ft_tp300 bash train_scripts/v8_macro/retrain.sh
#
# ``WARM_START=0`` trains from scratch (then give it a real number of epochs).
set -uo pipefail
cd "${RLNN_ROOT:-/home/hzeng/project/ZH/RLNN_lab}" || exit 1
source train_scripts/v8_macro/common.sh

export CUDA_VISIBLE_DEVICES="${V8_GPU:-1}"
mkdir -p "$OUT/queue"

RUN_NAME="${RUN_NAME:-ft_tp${NUM_TP:-300}}"
NUM_TP="${NUM_TP:-300}"
EPOCHS="${EPOCHS:-5}"
GRAPHS="${GRAPHS:-1000}"
LR="${LR:-1e-4}"
FINETUNE_LR="${FINETUNE_LR:-2e-5}"
BATCH="${BATCH:-64}"
GRAPH_BATCH="${GRAPH_BATCH:-8}"
NUM_K="${NUM_K:-20}"
NUM_H="${NUM_H:-128}"
NUM_L="${NUM_L:-5}"
LAMBD="${LAMBD:-0.5}"
WARM_START="${WARM_START:-1}"

schedule=()
if [ "$WARM_START" = 1 ]; then
  schedule=(--warm_start "$ROOT/outputs/v2/none_seed0/checkpoints/best.pt" --finetune_lr "$FINETUNE_LR")
fi

"$PYTHON" -u "$ROOT/tools/run_rlnn.py" train --version v2 --run "$RUN_NAME" \
  --frequency none --seed 0 --epochs "$EPOCHS" \
  --graph_batch_size "$GRAPH_BATCH" --batch_size "$BATCH" --lr "$LR" \
  --validation_seed 1017 \
  --train_path "$ROOT/models/data/MIS_tensors/RB_small_train" --train_samples "$GRAPHS" \
  --valid_path "$ROOT/models/data/MIS_tensors/RB_small_val" --valid_samples 100 \
  --test_path "$ROOT/models/data/author_rld4co/MIS/RB_small_test" \
  --num_workers "${WORKERS:-4}" \
  --num_k "$NUM_K" --num_t 100 --num_tp "$NUM_TP" --num_d 5 --num_l "$NUM_L" --num_h "$NUM_H" \
  --mixed_precision no --beta 1.02 --lambd "$LAMBD" \
  "${schedule[@]}" "$@" 2>&1 | tee "$OUT/queue/retrain_${RUN_NAME}.log"
