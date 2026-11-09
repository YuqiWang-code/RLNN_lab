#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
arm="${1:?pass baseline or a v6 arm}"
shift
case "$arm" in
  baseline) checkpoint="$(base_checkpoint "$REF_SEED")" ;;
  *) checkpoint="$OUT/train_${arm}_seed${SEED}/checkpoints" ;;
esac
exec "$PYTHON" -u "$ROOT/models/diagnose_v6_frequency.py" --arm "$arm" \
  --checkpoint "${CHECKPOINT_DIR:-$checkpoint}" --output "$OUT/${RUN_NAME:-diagnose_${arm}_seed${SEED}}" "$@"
