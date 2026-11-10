#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
protocol="${1:?pass t100, t1000, budget05 or budget10}"
shift
exec "$PYTHON" -u models/eval_static_prior.py --protocol "$protocol" --root "$OUT" \
  --output "$OUT/${RUN_NAME:-${protocol}_ref${REF_SEED}}" --ref "$REF_SEED" "$@"
