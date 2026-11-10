#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
policy="${1:?pass pave or random}"
gpu="${2:?pass GPU}"
export CUDA_VISIBLE_DEVICES="$gpu"
args=()
horizon=2
if [[ "$policy" == pave ]]; then
  horizon=1
  args=(--value-checkpoint "$OUT/stats_h1_seed0/checkpoints/best.pt")
fi
for reserve in 0.02 0.05 0.10; do
  bash train_scripts/legacy/v3/eval.sh --split validation --test_samples 100 \
    --save_dir "$ROOT/outputs/v2/none_seed0/checkpoints" --train_seed 0 --label RLNN-tuned \
    --post "$policy" --horizon "$horizon" --time-budget-s 0.5 --post-reserve-s "$reserve" \
    --cohort-id "reserve_${policy}_${reserve}" --result_dir "$OUT/reserve_${policy}_${reserve}" \
    "${args[@]}" > "$OUT/queue/reserve_${policy}_${reserve}.log" 2>&1
done
