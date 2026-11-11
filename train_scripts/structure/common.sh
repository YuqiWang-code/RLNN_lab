#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=backend:cudaMallocAsync
export RLNN_SOURCE_SHA="${RLNN_SOURCE_SHA:-$(date -u +%Y%m%dT%H%M%SZ)-unversioned}"
export RLNN_SOURCE_BRANCH="${RLNN_SOURCE_BRANCH:-structure-v1}"

PYTHON="${PYTHON:-/home/hzeng/envs/rlnn/bin/python}"
SCRIPTS="$ROOT/train_scripts/structure"
OUT="${OUT_V2:-$ROOT/outputs/structure_v1}"
DATA="$ROOT/models/data/MIS_tensors"
REFERENCE_SEED="${REFERENCE_SEED:-0}"
seeds=(42 123 999)
TRAIN_SEED="${TRAIN_SEED:-${seeds[$REFERENCE_SEED]}}"
ARMS=(degree gate gain all degree_out all_out)
SEARCH_SEED="${SEARCH_SEED:-1017}"

v1_args() {
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
