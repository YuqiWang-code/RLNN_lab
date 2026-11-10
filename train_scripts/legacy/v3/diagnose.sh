#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec "$PYTHON" -u models/diagnose_v3.py --starts \
  "$OUT/starts_validation_seed0_stream0/starts.pt" \
  "$OUT/starts_validation_seed1_stream0/starts.pt" \
  "$OUT/starts_validation_seed2_stream0/starts.pt" \
  --mode census --workers "${WORKERS:-24}" --result-dir "$OUT/census" "$@"
