#!/usr/bin/env bash
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHON="${PYTHON:-/home/hzeng/envs/rlnn_mamba/bin/python}"
SCRIPTS="$ROOT/train_scripts/mamba/v0"
OUT="$ROOT/outputs/v7_context"
DATA="$ROOT/models/data/MIS_tensors"
REF_SEED="${REF_SEED:-0}"
seeds=(42 123 999)
SEED="${seeds[$REF_SEED]}"
