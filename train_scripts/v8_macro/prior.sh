#!/usr/bin/env bash
# P0-B / P0-C: is an h-step macro transition worth modelling at all?
# Frozen policy, frozen PAVE, no training.  ~15 min on 60 validation graphs.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
mkdir -p "$OUT/prior"
"$PYTHON" -u models/diagnose_v8_prior.py --reference_seed "$REFERENCE_SEED" \
  --save_dir "$BACKBONE" --result_dir "$OUT" --split validation \
  --graphs "${GRAPHS:-60}" --steps 25 50 75 90 --particles 2 --repeats 24 \
  --macro_h 1 2 --shuffle_seeds 0 1 2 2>&1 | tee "$OUT/queue/prior_ref${REFERENCE_SEED}.log"
