#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec "$PYTHON" -u models/diagnose_v7_context.py --ref "$REF_SEED" \
  --output "$OUT/${RUN_NAME:-preflight/permutation}" "$@"
