#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
arm="${1:?pass a v7 context mode}"
shift
export PYTORCH_ALLOC_CONF=backend:cudaMallocAsync
exec "$PYTHON" -u models/train_v7_context.py --context_mode "$arm" --reference_seed "$REF_SEED" \
  --result_dir "$OUT/${RUN_NAME:-train_${arm}_ref${REF_SEED}}" --seed "$SEED" --epochs "${EPOCHS:-3}" \
  --v7_width "${WIDTH:-32}" \
  --graph_batch_size 8 --batch_size 64 --lr 0.0003 --alpha_lr 0.001 \
  --method RLNN --loss erdoes --problem mis --beta 1.02 --lambd 0.5 \
  --train_path "$DATA/RB_small_train" --train_samples 1000 \
  --valid_path "$DATA/RB_small_val" --valid_samples 100 \
  --test_path "$DATA/RB_small_author_test" --num_workers 2 \
  --num_k 20 --num_t 100 --num_tp 50 --num_d 5 --num_l 5 --num_h 128 --mixed_precision no "$@"
