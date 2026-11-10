#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
v2_args "$@"
split=""
for arg in "${EXTRA[@]}"; do
  [[ "$arg" != --train-targets ]] || split="-train"
done
exec "$PYTHON" -u models/diagnose_static_leverage.py --prepare --ref "$REF_SEED" --backend "$BACKEND" \
  --output "$OUT/oracles${split}_ref${REF_SEED}_${BACKEND}" "${EXTRA[@]}"
