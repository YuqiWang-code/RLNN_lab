#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
export PYTORCH_ALLOC_CONF=backend:cudaMallocAsync
arm="${1:?pass a v6 arm}"
shift
exec "$PYTHON" -u "$ROOT/models/train_v6_frequency.py" \
  --result_dir "$OUT/${RUN_NAME:-train_${arm}_seed${SEED}}" --frequency "$arm" --seed "$SEED" \
  --teacher_checkpoint "$(base_checkpoint "$REF_SEED")/best.pt" --epochs "${EPOCHS:-3}" \
  --graph_batch_size 8 --batch_size 64 --lr 0.0003 --validation_seed 1017 \
  --method RLNN --loss erdoes --problem mis --beta 1.02 --lambd 0.5 \
  --train_path "$DATA/MIS_tensors/RB_small_train" --train_samples 1000 \
  --valid_path "$DATA/MIS_tensors/RB_small_val" --valid_samples 100 \
  --test_path "$DATA/MIS_tensors/RB_small_author_test" --num_workers 2 \
  --num_k 20 --num_t 100 --num_tp 50 --num_d 5 --num_l 5 --num_h 128 --mixed_precision no "$@"
