#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
protocol="${1:?pass curve, budget or diagnose}"
shift
v2_args "$@"
oracles="${ORACLES:-$OUT/oracles_ref${REF_SEED}_${BACKEND}}"
common=(--ref "$REF_SEED" --backend "$BACKEND" --output "$OUT/p0_${protocol}_ref${REF_SEED}_${BACKEND}"
        --oracle-kamis-path "$oracles/oracle-kamis.json" --oracle-b-path "$oracles/oracle-b.json")
case "$protocol" in
  curve|budget) exec "$PYTHON" -u models/eval_static_prior.py v2 --protocol "$protocol" --repeats 1 "${common[@]}" "${EXTRA[@]}" ;;
  diagnose) exec "$PYTHON" -u models/diagnose_static_leverage.py "${common[@]}" "${EXTRA[@]}" ;;
  *) echo "Use curve, budget or diagnose" >&2; exit 2 ;;
esac
