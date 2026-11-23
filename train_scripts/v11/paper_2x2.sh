#!/usr/bin/env bash
set -euo pipefail

ROOT=${RLNN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT=${1:?usage: paper_2x2.sh OUTPUT_DIR [GPU]}
gpu=${2:-1}
cd "$ROOT"

seeds=(1017 1019 1021)
orders=(
    "eager_no_compile eager full no_compile"
    "no_compile full eager eager_no_compile"
    "eager full eager_no_compile no_compile"
)

for mode in fixed130 deadline200; do
    steps=0
    if [[ $mode == fixed130 ]]; then steps=130; fi
    for index in "${!seeds[@]}"; do
        seed=${seeds[$index]}
        read -r -a arms <<< "${orders[$index]}"
        if [[ $mode == deadline200 ]]; then
            arms=("${arms[3]}" "${arms[2]}" "${arms[1]}" "${arms[0]}")
        fi
        for arm in "${arms[@]}"; do
            printf '%s %s %s seed=%s\n' "$(date -Is)" "$mode" "$arm" "$seed"
            GRAPHS=1000 NUM_K=200 BUDGET=0.2 FIXED_STEPS="$steps" \
                CHECKPOINT_DIR=outputs/v2/none_seed0/checkpoints \
                RESULT_DIR="$OUT/$mode/$arm/batch_$seed" \
                bash "${PAPER_ABLATION_ENTRY:-train_scripts/v11/paper_ablation.sh}" "$arm" "$gpu" "$seed"
        done
    done
done
