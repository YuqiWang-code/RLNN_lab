#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=backend:cudaMallocAsync
export RLNN_SOURCE_SHA="${RLNN_SOURCE_SHA:-$(date -u +%Y%m%dT%H%M%SZ)-unversioned}"
export RLNN_SOURCE_BRANCH="${RLNN_SOURCE_BRANCH:-v7-tccr}"

PYTHON="${PYTHON:-/home/hzeng/envs/rlnn/bin/python}"
SCRIPTS="$ROOT/train_scripts/v7"
OUT="${OUT_V7:-$ROOT/outputs/v7_tccr}"
DATA="$ROOT/models/data/MIS_tensors"
REFERENCE_SEED="${REFERENCE_SEED:-0}"
SAMPLES="${SAMPLES:-100}"
SEEDS="${SEEDS:-1017 1019 1021}"

backbone_dir() {
  echo "$ROOT/outputs/v2/none_seed${REFERENCE_SEED}/checkpoints"
}

v7_args() {
  EXTRA=()
  while (($#)); do
    if [[ "$1" == --gpu ]]; then
      export CUDA_VISIBLE_DEVICES="${2:?pass one GPU index}"
      shift 2
    else
      EXTRA+=("$1")
      shift
    fi
  done
}
