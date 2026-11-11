#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

arm="${1:?pass one of: degree gate gain all}"
shift
v1_args "$@"
if [[ ! " ${ARMS[*]} " == *" $arm "* ]]; then
  echo "unknown structure arm $arm (use: ${ARMS[*]})" >&2
  exit 2
fi

mkdir -p "$OUT"
RUN="${OUT}/train_${arm}_ref${REFERENCE_SEED}"
"$PYTHON" -u models/train_structure.py \
  --structure_mode "$arm" \
  --reference_seed "$REFERENCE_SEED" --result_dir "$RUN" --seed "$TRAIN_SEED" \
  --epochs "${EPOCHS:-3}" --graph_batch_size 8 --batch_size 64 --lr "${LR:-0.0005}" \
  --method RLNN --loss erdoes --problem mis --beta 1.02 --lambd "${LAMBD:-0.5}" \
  --train_path "$DATA/RB_small_train" --train_samples "${TRAIN_SAMPLES:-1000}" \
  --valid_path "$DATA/RB_small_val" --valid_samples 100 --validation_offset 0 \
  --test_path "$DATA/RB_small_author_test" --num_workers 2 \
  --num_k 20 --num_t 100 --num_tp 50 --num_d 5 --num_l 5 --num_h 128 --mixed_precision no \
  "${EXTRA[@]}" 2>&1 | tee "${OUT}/train_${arm}_ref${REFERENCE_SEED}.log"
