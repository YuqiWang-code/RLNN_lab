#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
arm="${1:?pass none or a v4 frequency arm}"
shift
seed="${SEED:-0}"
checkpoint="$OUT/${TRAIN_RUN:-${arm}_seed${seed}}/checkpoints"
if [[ "$arm" == none && -z "${TRAIN_RUN:-}" ]]; then checkpoint="$(base_checkpoint "$seed")"; fi
exec "$PYTHON" -u "$ROOT/models/diagnose_v4_frequency.py" \
  --frequency "$arm" --save_dir "${CHECKPOINT:-$checkpoint}" \
  --result_dir "$OUT/${RUN_NAME:-diagnostic_${arm}_seed${seed}_${PRECISION:-fp16}}" \
  --label "${LABEL:-$arm}" --train_seed "$seed" --seed 1017 \
  --split "${SPLIT:-validation}" --test_samples "${SAMPLES:-100}" \
  --mixed_precision "${PRECISION:-fp16}" --post none pave \
  --diagnostic-particles "${DIAGNOSTIC_PARTICLES:-4}" --spectral-graphs "${SPECTRAL_GRAPHS:-20}" \
  --horizon 1 --num_k 20 --num_t 100 --num_d 5 --num_l 5 --num_h 128 \
  --method RLNN --problem mis --beta 1.02 --num_workers 0 "$@"
