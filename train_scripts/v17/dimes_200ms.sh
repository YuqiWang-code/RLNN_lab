#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
split=${1:?expected graph split}
checkpoint=${2:?expected exported DIMES-MIS weights}
out=${3:?expected output directory}
case "$split" in
    RB_small|RB_large) graphs=1000 ;;
    ER_small) graphs=128 ;;
    ER_large) graphs=16 ;;
    *) echo "Unknown graph split: $split" >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/models${PYTHONPATH:+:$PYTHONPATH}"
exec taskset -c "${CPU_CORE:-11}" "${PYTHON:-$HOME/envs/rlnn/bin/python}" -u "${REFERENCE_ENTRY:-models/eval_reference_deadline.py}" dimes \
    --dataset "models/data/MIS_tensors/${split}_author_test" --graphs "${4:-$graphs}" --seeds 1017 1019 1021 \
    --checkpoint "$checkpoint" --output "$out"
