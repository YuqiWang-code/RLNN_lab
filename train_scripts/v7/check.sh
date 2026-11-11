#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

mkdir -p "$OUT"

echo "== operator checks (CPU, no checkpoints) =="
"$PYTHON" models/check_v7.py

echo "== capture parity against eval_v2.search =="
"$PYTHON" -u models/diagnose_v7_census.py \
  --reference_seed "$REFERENCE_SEED" --save_dir "$(backbone_dir)" --result_dir "$OUT/checks_smoke" \
  --split validation --test_samples 3 --seeds 1017 --phase both --check_search --check_graphs 3 \
  2>&1 | tee "$OUT/checks_smoke.log"
