#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
export PYTHONPATH="$ROOT/models${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -u models/archived_frequency/diagnose_v5_frequency.py --seed "${SEED:-0}" \
  --split "${SPLIT:-train}" --samples "${SAMPLES:-20}" --precision "${PRECISION:-fp32}" \
  --output "$OUT/${RUN_NAME:-diagnostics_${SPLIT:-train}_seed${SEED:-0}_${PRECISION:-fp32}}" "$@"
