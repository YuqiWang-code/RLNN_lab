#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

phase="${1:?pass recombine, conflict or both}"
shift || true
v7_args "$@"

mkdir -p "$OUT"
log="$OUT/census_ref${REFERENCE_SEED}_${phase}.log"
args=(--reference_seed "$REFERENCE_SEED" --save_dir "$(backbone_dir)" --result_dir "$OUT"
      --split validation --test_samples "$SAMPLES" --phase "$phase" --seeds $SEEDS)
[[ -n "${CUDA_GRAPH:-}" ]] && args+=(--cuda-graph)

"$PYTHON" -u models/diagnose_v7_census.py "${args[@]}" "${EXTRA[@]}" 2>&1 | tee "$log"
