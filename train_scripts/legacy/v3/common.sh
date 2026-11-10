#!/usr/bin/env bash
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=backend:native
PYTHON="${PYTHON:-/home/hzeng/envs/rlnn/bin/python}"
OUT="$ROOT/outputs/v3"
DATA="$ROOT/models/data"
base_checkpoint() {
  case "$1" in
    0) echo "$ROOT/outputs/v000_rlnn_baseline/rb_train_author_b64_g8_gpu1_seed0/checkpoints" ;;
    1|2) echo "$ROOT/outputs/v1/rlnn_seed$1/checkpoints" ;;
    *) return 1 ;;
  esac
}
