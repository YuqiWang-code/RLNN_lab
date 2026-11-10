#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
SEED="${SEED:-0}"
STREAM="${STREAM:-0}"
SPLIT="${SPLIT:-validation}"
CHECKPOINT="${CHECKPOINT:-$(base_checkpoint "$SEED")}"
NAME="${NAME:-starts_${SPLIT}_seed${SEED}_stream${STREAM}}"
COUNT="${COUNT:-100}"
exec "$PYTHON" -u models/collect_v3_starts.py --data_split "$SPLIT" --stream "$STREAM" \
  --save_dir "$CHECKPOINT" --train_seed "$SEED" --test_samples "$COUNT" \
  --result_dir "$OUT/$NAME" "$@"
