#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
arm="${1:?pass baseline, v5_control or a v6 arm}"
shift
case "$arm" in
  baseline) frequency=none; checkpoint="$(base_checkpoint "$REF_SEED")"; train_seed="$REF_SEED" ;;
  v5_control)
    frequency="${V5_MODE:-v5a}"
    checkpoint="$ROOT/outputs/v5_frequency/train_${frequency}_seed${REF_SEED}/checkpoints"
    train_seed="$REF_SEED" ;;
  *) frequency="$arm"; checkpoint="$OUT/train_${arm}_seed${SEED}/checkpoints"; train_seed="$SEED" ;;
esac
exec "$PYTHON" -u "$ROOT/models/eval_v6_frequency.py" \
  --result_dir "$OUT/${RUN_NAME:-eval_${arm}_seed${SEED}}" --label "$arm" --frequency "$frequency" \
  --reference-seed "$REF_SEED" --train_seed "$train_seed" --save_dir "${CHECKPOINT_DIR:-$checkpoint}" "$@"
