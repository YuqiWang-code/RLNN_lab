#!/usr/bin/env bash
set -euo pipefail

ROOT=${RLNN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$ROOT"
variant=${1:?usage: paper_ablation.sh VARIANT [GPU] [SEED ...]}
gpu=${2:-1}
seeds=(1017 1019 1021)
if (( $# > 2 )); then seeds=("${@:3}"); fi

PYTHON=${PYTHON:-"$HOME/envs/rlnn/bin/python"}
EVAL_ENTRY=${EVAL_ENTRY:-models/eval_v8_padded.py}
OUT=${RESULT_DIR:-"outputs/v11/paper_ablations/$variant"}
export OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/models${PYTHONPATH:+:$PYTHONPATH}"

backend=captured
precision=fp16
decode_top=4
particles=${NUM_K:-200}
schedule=midpoint
leading=(--skip_incumbent)
compile=(--compile_model)
adjacency=(--dense)
post=(--no_pave)
reserve_ms=${PAVE_RESERVE_MS:-40}
case "$variant" in
    full) ;;
    eager) backend=eager ;;
    no_compile) compile=() ;;
    eager_no_compile) backend=eager; compile=() ;;
    csr) adjacency=() ;;
    csr_no_compile) adjacency=(); compile=() ;;
    fp32) precision=no ;;
    decode_all) decode_top=$particles ;;
    leading) leading=() ;;
    terminal) schedule=terminal ;;
    pave) post=(--reserve_ms "$reserve_ms") ;;
    holdout) post=(--no_pave --reserve_ms "$reserve_ms") ;;
    *) echo "Unknown variant: $variant" >&2; exit 2 ;;
esac

common=(--split test --test_path models/data/MIS_tensors/RB_small_author_test
        --test_samples "${GRAPHS:-1000}" --reference_seed 0
        --save_dir "${CHECKPOINT_DIR:-outputs/v2/none_seed0/checkpoints}" --num_k "$particles" --chunk 10
        --decoders captured --decode_every 1000 --decode_top "$decode_top"
        --mixed_precision "$precision" --repeats 1 --budgets "${BUDGET:-0.2}"
        --fixed_steps "${FIXED_STEPS:-0}" --search_backend "$backend" --decode_schedule "$schedule")

mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
for seed in "${seeds[@]}"; do
    dir="$OUT/seed_$seed"
    mkdir "$dir"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$EVAL_ENTRY" "${common[@]}" "${leading[@]}" \
        "${compile[@]}" "${adjacency[@]}" "${post[@]}" --seeds "$seed" \
        --result_dir "$dir" --v11_csv "$dir/v11_rows.csv" 2>&1 | tee "$dir/run.log"
done
