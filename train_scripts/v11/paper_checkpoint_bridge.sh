#!/usr/bin/env bash
set -euo pipefail

ROOT=${RLNN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT=${1:?usage: paper_checkpoint_bridge.sh OUTPUT_DIR [GPU]}
gpu=${2:-1}
cd "$ROOT"
for seed in 1017 1019 1021; do
    counts=(20 200)
    if [[ $seed == 1019 ]]; then counts=(200 20); fi
    for k in "${counts[@]}"; do
        GRAPHS=1000 NUM_K="$k" BUDGET=0.2 FIXED_STEPS=0 \
            CHECKPOINT_DIR=models/data/author_rld4co/checkpoints/erdoes/rlnn_rb_small_mis \
            RESULT_DIR="$OUT/author_k$k/batch_$seed" \
            bash "${PAPER_ABLATION_ENTRY:-train_scripts/v11/paper_ablation.sh}" full "$gpu" "$seed"
    done
done
