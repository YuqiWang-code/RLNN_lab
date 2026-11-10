#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

seed="${SEED:-0}"
run="${RUN_NAME:-rlnn_seed${seed}}"
test="$ROOT/models/data/author_rld4co/MIS/RB_small_test"
exec "${PYTHON:-python}" "$ROOT/tools/run_rlnn.py" train --version v1 --run "$run" \
  --seed "$seed" --epochs 50 --graph_batch_size 8 --batch_size 64 --lr 0.0001 \
  --train_path "$ROOT/models/data/MIS_tensors/RB_small_train" --train_samples 1000 \
  --valid_path "$ROOT/models/data/MIS_tensors/RB_small_val" --valid_samples 100 \
  --test_path "$test" --num_workers 8 --num_l 5 --num_h 128 --mixed_precision no \
  --num_k 20 --num_t 100 --num_tp 50 --num_d 5 --beta 1.02 --lambd 0.5 "$@"
