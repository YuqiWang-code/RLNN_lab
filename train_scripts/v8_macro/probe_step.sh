#!/usr/bin/env bash
# Where does the per-step time go in the deployment (budget) path?
# Pure measurement: same checkpoint, same graph, eager vs CUDA-graph-captured step.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

mkdir -p "$OUT/queue"
"$PYTHON" -u models/profile_budget_step.py \
  --reference_seed "$REFERENCE_SEED" --save_dir "$BACKBONE" --split validation \
  --graphs 3 --steps 200 --repeat 3 \
  --result_dir "$OUT/profile_step" 2>&1 | tee -a "$OUT/queue/profile_step.log"
