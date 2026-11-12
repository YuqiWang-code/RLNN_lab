#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
arm="${1:?pass A2, A3, A5}"
mkdir -p "$OUT"
"$PYTHON" -u models/v8_train.py --arm "$arm" --reference_seed "$REFERENCE_SEED" \
  --train_blocks "$OUT/data/blocks_train_ref${REFERENCE_SEED}.pt" --backbone "$BACKBONE/best.pt" \
  --save_dir "$OUT/train_${arm}_ref${REFERENCE_SEED}" --epochs "${EPOCHS:-3}" 2>&1 \
  | tee "$OUT/train_${arm}_ref${REFERENCE_SEED}.log"
