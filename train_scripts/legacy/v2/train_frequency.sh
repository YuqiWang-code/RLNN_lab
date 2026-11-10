#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
export PYTORCH_ALLOC_CONF=backend:cudaMallocAsync
arm="${1:?pass a frequency arm or none}"
shift
seed="${SEED:-0}"
run="${arm}_seed${seed}"
schedule=(--warm_start "$(base_checkpoint "$seed")/best.pt" --finetune_lr 0.00002)
if [[ "$arm" != none ]]; then schedule+=(--freeze_backbone_epochs 1); fi
epochs=3
if [[ "${FROM_SCRATCH:-0}" == 1 ]]; then
  schedule=()
  epochs=50
  run="scratch_$run"
fi
exec "$PYTHON" "$ROOT/tools/run_rlnn.py" train --version v2 --run "${RUN_NAME:-$run}" \
  --frequency "$arm" --seed "$seed" --epochs "$epochs" --graph_batch_size 8 --batch_size 64 \
  --lr 0.0001 --validation_seed 1017 "${schedule[@]}" \
  --train_path "$DATA/MIS_tensors/RB_small_train" --train_samples 1000 \
  --valid_path "$DATA/MIS_tensors/RB_small_val" --valid_samples 100 \
  --test_path "$DATA/author_rld4co/MIS/RB_small_test" --num_workers 2 \
  --num_k 20 --num_t 100 --num_tp 50 --num_d 5 --num_l 5 --num_h 128 \
  --mixed_precision no --beta 1.02 --lambd 0.5 "$@"
