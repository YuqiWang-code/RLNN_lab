#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export RLNN_SOURCE_SHA="${RLNN_SOURCE_SHA:-$(date -u +%Y%m%dT%H%M%SZ)-unversioned}"
export RLNN_SOURCE_BRANCH="${RLNN_SOURCE_BRANCH:-v8-macro}"
PYTHON="${PYTHON:-/home/hzeng/envs/rlnn/bin/python}"
OUT="${OUT_V8:-$ROOT/outputs/v8_macro}"
REFERENCE_SEED="${REFERENCE_SEED:-0}"
SEEDS="${SEEDS:-1017 1019 1021}"
HORIZON="${HORIZON:-2}"
BACKBONE="$ROOT/outputs/v2/none_seed${REFERENCE_SEED}/checkpoints"
