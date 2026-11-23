#!/usr/bin/env bash
set -euo pipefail

ROOT=${RLNN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT=${1:?usage: paper_precision.sh OUTPUT_DIR [GPU]}
gpu=${2:-1}
cd "$ROOT"
export NVIDIA_TF32_OVERRIDE=0

for mode in fixed130 deadline200; do
    steps=0
    if [[ $mode == fixed130 ]]; then steps=130; fi
    for seed in 1017 1019 1021; do
        arms=(fp16 fp32)
        if [[ $seed == 1019 ]]; then arms=(fp32 fp16); fi
        if [[ $mode == deadline200 ]]; then arms=("${arms[1]}" "${arms[0]}"); fi
        for arm in "${arms[@]}"; do
            variant=$arm
            if [[ $arm == fp16 ]]; then variant=full; fi
            printf '%s %s %s seed=%s\n' "$(date -Is)" "$mode" "$arm" "$seed"
            GRAPHS=1000 NUM_K=200 BUDGET=0.2 FIXED_STEPS="$steps" \
                CHECKPOINT_DIR=outputs/v2/none_seed0/checkpoints \
                RESULT_DIR="$OUT/$mode/$arm/batch_$seed" \
                bash "${PAPER_ABLATION_ENTRY:-train_scripts/v11/paper_ablation.sh}" "$variant" "$gpu" "$seed"
        done
    done
done
