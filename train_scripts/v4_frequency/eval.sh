#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
arm="${1:?pass none or a v4 frequency arm}"
shift
seed="${SEED:-0}"
split="${SPLIT:-test}"
samples="${SAMPLES:-1000}"
if [[ "$split" == validation && -z "${SAMPLES:-}" ]]; then samples=100; fi
checkpoint="$OUT/${TRAIN_RUN:-${arm}_seed${seed}}/checkpoints"
if [[ "$arm" == none && -z "${TRAIN_RUN:-}" ]]; then checkpoint="$(base_checkpoint "$seed")"; fi
exec "$PYTHON" -u "$ROOT/models/eval_v4_frequency.py" \
  --frequency "$arm" --save_dir "${CHECKPOINT:-$checkpoint}" \
  --result_dir "$OUT/${RUN_NAME:-eval_${arm}_seed${seed}_${split}_${PRECISION:-fp16}}" \
  --label "${LABEL:-$arm}" --train_seed "$seed" --seed 1017 \
  --split "$split" --test_samples "$samples" --mixed_precision "${PRECISION:-fp16}" \
  --post none pave --horizon 1 --num_k 20 --num_t 100 --num_d 5 \
  --num_l 5 --num_h 128 --method RLNN --problem mis --beta 1.02 \
  --num_workers 0 "$@"
