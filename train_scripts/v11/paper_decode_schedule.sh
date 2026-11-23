#!/usr/bin/env bash
set -euo pipefail

ROOT=${RLNN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT=${1:?usage: paper_decode_schedule.sh OUTPUT_DIR [GPU] [terminal|decode_all]}
gpu=${2:-1}
variant=${3:-terminal}
case "$variant" in
    terminal|decode_all) ;;
    *) echo "Expected terminal or decode_all" >&2; exit 2 ;;
esac
cd "$ROOT"

for seed in 1017 1019 1021; do
    arms=(full "$variant")
    if [[ $seed == 1019 ]]; then arms=("$variant" full); fi
    for arm in "${arms[@]}"; do
        printf '%s %s seed=%s\n' "$(date -Is)" "$arm" "$seed"
        GRAPHS=1000 NUM_K=200 BUDGET=0.2 FIXED_STEPS=0 \
            CHECKPOINT_DIR=outputs/v2/none_seed0/checkpoints \
            RESULT_DIR="$OUT/$arm/batch_$seed" \
            bash "${PAPER_ABLATION_ENTRY:-train_scripts/v11/paper_ablation.sh}" "$arm" "$gpu" "$seed"
    done
done
