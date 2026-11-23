#!/usr/bin/env bash
# Paired 0.2 s runs on the author test set, on one GPU.
set -euo pipefail

ROOT=${RLNN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$ROOT"
PYTHON=${PYTHON:-"$HOME/envs/rlnn/bin/python"}
EVAL_ENTRY=${EVAL_ENTRY:-models/eval_v8_padded.py}
OUT=${RESULT_DIR:-outputs/v11/fused_pave_02s_deadline}
GRAPHS=${GRAPHS:-1000}
GPU=${1:-1}
if (( $# > 0 )); then shift; fi
seeds=(1017 1019 1021)
if (( $# > 0 )); then seeds=("$@"); fi
export OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/models${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
common=(--split test --test_path models/data/MIS_tensors/RB_small_author_test
        --test_samples "$GRAPHS" --reference_seed 0 --save_dir outputs/v2/none_seed0/checkpoints
        --num_k 200 --decoders captured --decode_every 1000 --decode_top 4
        --dense --mixed_precision fp16 --repeats 1 --budgets 0.2 --skip_incumbent --compile_model)

run_arm() {
    local seed=$1 arm=$2
    local dir="$OUT/${arm}_s${seed}"
    local flags=()
    if [[ $arm == pave_off ]]; then flags+=(--no_pave); fi
    mkdir "$dir"
    echo "$(date -Is) $arm seed $seed"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u "$EVAL_ENTRY" "${common[@]}" "${flags[@]}" \
        --seeds "$seed" --result_dir "$dir" --v11_csv "$dir/v11_rows.csv" > "$dir/run.log" 2>&1
    tail -n 2 "$dir/run.log"
}

for seed in "${seeds[@]}"; do
    if [[ $seed == 1019 ]]; then
        run_arm "$seed" pave_off
        run_arm "$seed" pave_on
    else
        run_arm "$seed" pave_on
        run_arm "$seed" pave_off
    fi
done
