#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
arm="${1:?pass baseline or a v7 context arm}"
shift
if [[ "$arm" == baseline ]]; then
  mode=none
  checkpoint="$ROOT/outputs/v2/none_seed${REF_SEED}/checkpoints"
  model_seed="$REF_SEED"
else
  mode="$arm"
  checkpoint="$OUT/train_${arm}_ref${REF_SEED}/checkpoints"
  model_seed="$SEED"
fi
exec "$PYTHON" -u models/eval_v7_context.py --label "$arm" --context_mode "$mode" \
  --v7_width "${WIDTH:-32}" \
  --reference_seed "$REF_SEED" --train_seed "$model_seed" --save_dir "${CHECKPOINT_DIR:-$checkpoint}" \
  --result_dir "$OUT/${RUN_NAME:-fixed_${arm}_ref${REF_SEED}}" \
  --split "${SPLIT:-validation}" --test_samples "${SAMPLES:-100}" "$@"
