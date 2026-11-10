#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
stage="${1:?pass p1, oracle or p2}"
protocol="${2:?pass curve or budget}"
shift 2
v2_args "$@"
case "$stage" in
  p1|oracle)
    extra=(--checkpoint "$OUT/${stage}_centered8_ref${REF_SEED}"
           --arms b0 null-rand centered masked matched-rand)
    if [[ "$stage" == p1 ]]; then
      extra=(--checkpoint "$OUT/p1_centered8_ref${REF_SEED}" --arms b0 null-rand centered masked matched-rand centered5 constants
             --compare centered5 "$OUT/p1_centered5_ref${REF_SEED}" --compare constants "$OUT/p1_constants_ref${REF_SEED}")
    fi
    ;;
  p2)
    extra=(--checkpoint "$OUT/p2_mamba8_ref${REF_SEED}"
           --arms b0 null-rand mamba8 masked matched-rand mlp8 msg8 mamba8-shuffle mamba8-degree
           --compare mamba8 "$OUT/p2_mamba8_ref${REF_SEED}"
           --compare mlp8 "$OUT/p2_mlp8_ref${REF_SEED}" --compare msg8 "$OUT/p2_msg8_ref${REF_SEED}"
           --compare mamba8-shuffle "$OUT/p2_mamba8-shuffle_ref${REF_SEED}"
           --compare mamba8-degree "$OUT/p2_mamba8-degree_ref${REF_SEED}")
    ;;
  *) echo "Use p1, oracle or p2" >&2; exit 2 ;;
esac
exec "$PYTHON" -u models/eval_static_prior.py v2 --ref "$REF_SEED" --backend "$BACKEND" --protocol "$protocol" \
  --output "$OUT/${stage}_${protocol}_ref${REF_SEED}_${BACKEND}" "${extra[@]}" "${EXTRA[@]}"
