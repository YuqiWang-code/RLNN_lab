#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a directory containing best.pt}"
cd "$ROOT/models"
exec "${PYTHON:-python}" "$ROOT/tools/run_rlnn.py" eval --run "${RUN_NAME:-eval_$(date +%Y%m%d_%H%M%S)}" \
  --checkpoint "$CHECKPOINT_DIR" --test_path "${TEST_PATH:-$ROOT/models/data/MIS_tensors/RB_small_test}" \
  --num_workers "${NUM_WORKERS:-8}" --method RLNN --loss erdoes --num_l 5 --num_h 128 --mixed_precision fp16 "$@"
