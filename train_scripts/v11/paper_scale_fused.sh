#!/usr/bin/env bash
set -euo pipefail

ROOT=${RLNN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$ROOT"
split=${1:?usage: paper_scale_fused.sh SPLIT OUTPUT_DIR [GPU]}
out=${2:?usage: paper_scale_fused.sh SPLIT OUTPUT_DIR [GPU]}
gpu=${3:-1}
case "$split" in
    RB_large|ER_small|ER_large) ;;
    *) echo "Expected RB_large, ER_small, or ER_large" >&2; exit 2 ;;
esac
test_dir="models/data/MIS_tensors/${split}_author_test"
test -s "$test_dir/tensor_manifest.json"
mkdir -p "$(dirname "$out")"
mkdir "$out"
export OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/models${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON:-$HOME/envs/rlnn/bin/python}" -u \
    "${EVAL_ENTRY:-models/eval_v8_padded.py}" --split test --test_path "$test_dir" \
    --test_samples -1 --save_dir outputs/v2/none_seed0/checkpoints --reference_seed 0 \
    --num_k 200 --chunk 10 --decoders captured --decode_every 1000 --decode_top 4 \
    --mixed_precision fp16 --budgets 0.2 --repeats 1 --fixed_steps 0 --search_backend captured \
    --decode_schedule midpoint --skip_incumbent --compile_model --dense --no_pave \
    --seeds 1017 1019 1021 --result_dir "$out" --v11_csv "$out/per_graph.csv" 2>&1 | tee "$out/run.log"
