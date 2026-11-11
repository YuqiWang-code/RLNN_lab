#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

protocol="${1:?pass fixed, fixed1000, budget or baseline}"
arm="${2:?pass degree, gate, gain, all, or baseline}"
shift 2
v1_args "$@"

mkdir -p "$OUT"
case "$protocol" in
  fixed)     protocol_args=(--num_t 100);  result="$OUT/fixed_${arm}_ref${REFERENCE_SEED}_s${SEARCH_SEED}" ;;
  fixed1000) protocol_args=(--num_t 1000); result="$OUT/fixed1000_${arm}_ref${REFERENCE_SEED}_s${SEARCH_SEED}" ;;
  budget)    protocol_args=(--time_budget_s "${BUDGET:?set BUDGET=0.5 or 1.0}" --post pave
                            --extended_search --post_reserve_s 0.05)
             result="$OUT/budget${BUDGET//./}_${arm}_ref${REFERENCE_SEED}_s${SEARCH_SEED}" ;;
  *)         echo "unknown protocol $protocol (fixed | fixed1000 | budget)" >&2; exit 2 ;;
esac

if [[ "$protocol" == *baseline* || "$arm" == baseline ]]; then
  save_dir="$ROOT/outputs/v2/none_seed${REFERENCE_SEED}/checkpoints"
  mode=none
  label="baseline_ref${REFERENCE_SEED}"
else
  save_dir="$OUT/train_${arm}_ref${REFERENCE_SEED}/checkpoints"
  mode="$arm"
  label="${arm}_ref${REFERENCE_SEED}"
fi
[[ -d "$save_dir" ]] || { echo "missing checkpoint dir $save_dir" >&2; exit 2; }

python_args=(--structure_mode "$mode" --reference_seed "$REFERENCE_SEED" --save_dir "$save_dir"
             --label "$label" --result_dir "$result" --split validation --seed "$SEARCH_SEED" "${protocol_args[@]}")
[[ -n "${CUDA_GRAPH:-}" ]] && python_args+=(--cuda-graph)

"$PYTHON" -u models/eval_structure.py "${python_args[@]}" "${EXTRA[@]}" 2>&1 | tee "${result}.log"
