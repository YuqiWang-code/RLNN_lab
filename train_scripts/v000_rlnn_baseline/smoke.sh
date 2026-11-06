#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
cd "$ROOT/models"
exec "${PYTHON:-python}" "$ROOT/tools/run_rlnn.py" smoke --run "${RUN_NAME:-smoke_$(date +%Y%m%d_%H%M%S)}" \
  --method RLNN --loss erdoes --num_l 5 --num_h 128 "$@"
