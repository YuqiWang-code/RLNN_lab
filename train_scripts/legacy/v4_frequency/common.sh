#!/usr/bin/env bash
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/models${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHON="${PYTHON:-python}"
OUT="$ROOT/outputs/v4_frequency"
DATA="$ROOT/models/data"

base_checkpoint() {
  echo "$ROOT/outputs/v2/none_seed$1/checkpoints"
}

pave_checkpoint() {
  echo "$ROOT/outputs/v3/stats_h1_seed$1/checkpoints/best.pt"
}
