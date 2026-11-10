#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
mkdir -p "$OUT/preflight"
for device in cpu cuda; do
  "$PYTHON" train_scripts/legacy/v5_frequency/check_adapter.py --device "$device" --output "$OUT/preflight/$device.json"
done
EPOCHS=1 RUN_NAME=preflight/train bash train_scripts/legacy/v5_frequency/train.sh v5a --train_samples 8 --valid_samples 3
