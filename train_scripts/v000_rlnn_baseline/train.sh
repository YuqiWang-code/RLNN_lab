#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-backend:cudaMallocAsync}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
cd "$ROOT/models"
exec "${PYTHON:-python}" "$ROOT/tools/run_rlnn.py" train --run "${RUN_NAME:-train_$(date +%Y%m%d_%H%M%S)}" \
  --train_path "${TRAIN_PATH:-$ROOT/models/data/MIS_tensors/RB_small_train}" \
  --valid_path "${VALID_PATH:-$ROOT/models/data/MIS_tensors/RB_small_val}" \
  --test_path "${TEST_PATH:-$ROOT/models/data/MIS_tensors/RB_small_test}" \
  --num_workers "${NUM_WORKERS:-8}" --method RLNN --loss erdoes --num_l 5 --num_h 128 --batch_size "${BATCH_SIZE:-1024}" --graph_batch_size "${GRAPH_BATCH_SIZE:-13}" "$@"
