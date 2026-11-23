#!/usr/bin/env bash
set -euo pipefail

ROOT=${RLNN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT=${1:?usage: paper_decode_reserve.sh OUTPUT_DIR [GPU]}
gpu=${2:-1}
cd "$ROOT"
for seed in 1017 1019 1021; do
    arms=(leading holdout)
    if [[ $seed == 1019 ]]; then arms=(holdout leading); fi
    for arm in "${arms[@]}"; do
        GRAPHS=1000 NUM_K=200 BUDGET=0.2 FIXED_STEPS=0 PAVE_RESERVE_MS=40 \
            CHECKPOINT_DIR=outputs/v2/none_seed0/checkpoints RESULT_DIR="$OUT/$arm/batch_$seed" \
            bash "${PAPER_ABLATION_ENTRY:-train_scripts/v11/paper_ablation.sh}" "$arm" "$gpu" "$seed"
    done
done
