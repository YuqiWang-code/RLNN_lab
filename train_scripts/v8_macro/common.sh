#!/usr/bin/env bash
# Shared defaults for the v8 launchers.
#
# Launchers are normally run from disk (``bash train_scripts/v8_macro/x.sh``), but
# tmp/remote.py pipes them into ``bash -s``, where BASH_SOURCE is unset.  In that
# case the current directory is the repository root, which is all we need.
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
  ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
else
  ROOT="$PWD"
fi

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
