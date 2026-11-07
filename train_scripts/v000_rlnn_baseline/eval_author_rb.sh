#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=backend:cudaMallocAsync
version=v000_rlnn_baseline
decoder="${DECODER:-greedy}"
weights="${WEIGHTS:-epoch48}"
case "$weights" in
  epoch48)
    checkpoint="$ROOT/outputs/$version/rb_train_g13_cache_w8_gpu1_seed0/checkpoints"
    checkpoint_args=(--checkpoint_epoch 48)
    ;;
  author_erdoes)
    checkpoint="$ROOT/models/data/author_rld4co/checkpoints/erdoes/rlnn_rb_small_mis"
    checkpoint_args=(--checkpoint_source "https://huggingface.co/shengyuf/RLNN/resolve/0b96a439a11ac90dc69512720084d08784f02bf4/erdoes/rlnn_rb_small_mis/best.pt")
    ;;
  *)
    printf '%s\n' "Unknown WEIGHTS: $weights" >&2
    exit 2
    ;;
esac
run="$ROOT/outputs/$version/${RUN_NAME:-rb_author_test_gpu1_${weights}_${decoder}_$(date +%Y%m%d_%H%M%S)}"
test="$ROOT/models/data/author_rld4co/MIS/RB_small_test"
cd "$ROOT/models"
exec "${PYTHON:-python}" -u eval_author_rb.py --save_dir "$checkpoint" --test_path "$test" --result_dir "$run" --decoder "$decoder" "${checkpoint_args[@]}"
