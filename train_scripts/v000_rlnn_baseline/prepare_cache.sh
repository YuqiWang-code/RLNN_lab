#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
for split in train val test; do
  CUDA_VISIBLE_DEVICES='' "${PYTHON:-python}" models/data/prepare_mis_tensors.py \
    --source "$ROOT/models/data/MIS/RB_small_$split" \
    --destination "$ROOT/models/data/MIS_tensors/RB_small_$split" --workers "${PROCESSES:-8}"
done
