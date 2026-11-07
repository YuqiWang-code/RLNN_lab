#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run="${RUN_NAME:-smoke_$(date +%Y%m%d_%H%M%S)}"
exec "${PYTHON:-python}" "$ROOT/tools/run_rlnn.py" smoke --version v1 --run "$run" \
  --num_h 16 --num_l 2 --frequency dynamic
