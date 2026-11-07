#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=backend:cudaMallocAsync
version=v000_rlnn_baseline
run="$ROOT/outputs/$version/${RUN_NAME:-rb_author_test_gpu1_epoch48_$(date +%Y%m%d_%H%M%S)}"
checkpoint="$ROOT/outputs/$version/rb_train_g13_cache_w8_gpu1_seed0/checkpoints"
test="$ROOT/models/data/author_rld4co/MIS/RB_small_test"
cd "$ROOT/models"
exec "${PYTHON:-python}" -u eval_author_rb.py --save_dir "$checkpoint" --test_path "$test" --result_dir "$run"
