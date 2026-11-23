#!/usr/bin/env bash
set -euo pipefail

ROOT=${RLNN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT=${1:?usage: paper_online_panel.sh OUTPUT_DIR [GPU]}
gpu=${2:-1}
cd "$ROOT"

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$ROOT/models${PYTHONPATH:+:$PYTHONPATH}"
seeds=(1023 1025 1027)
mkdir -p "$OUT"
"${PYTHON:-$HOME/envs/rlnn/bin/python}" -u "${REFERENCE_ENTRY:-models/eval_reference_deadline.py}" kamis \
    --seeds "${seeds[@]}" --graphs 1000 --kamis tmp/kamis_build/online_mis \
    --output "$OUT/kamis_online" 2>&1 | tee "$OUT/kamis_online.log"
GRAPHS=1000 NUM_K=200 BUDGET=0.2 FIXED_STEPS=0 CHECKPOINT_DIR=outputs/v2/none_seed0/checkpoints \
    RESULT_DIR="$OUT/fused" bash "${PAPER_ABLATION_ENTRY:-train_scripts/v11/paper_ablation.sh}" \
    full "$gpu" "${seeds[@]}"
