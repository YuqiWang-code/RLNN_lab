#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

# End-to-end fixed-work comparison: A.3 vs recombination vs frozen PAVE vs both.
v7_args "$@"
mkdir -p "$OUT"
log="$OUT/pipeline_ref${REFERENCE_SEED}.log"
args=(--reference_seed "$REFERENCE_SEED" --save_dir "$(backbone_dir)" --result_dir "$OUT"
      --split validation --test_samples "$SAMPLES" --phase recombine --seeds $SEEDS)
[[ -n "${CUDA_GRAPH:-}" ]] && args+=(--cuda-graph)

"$PYTHON" -u models/eval_v7_tccr.py "${args[@]}" "${EXTRA[@]}" 2>&1 | tee "$log"
