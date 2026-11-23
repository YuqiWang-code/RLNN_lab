#!/usr/bin/env bash
set -euo pipefail
root=${RLNN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$root"
regime=${1:?usage: priority_200ms.sh REGIME ARM DATASET OUTPUT [SEED ...]}
arm=${2:?expected learned, degree, or neutral}
dataset=${3:?expected a directory containing tensor_manifest.json}
output=${4:?expected a new output directory}
shift 4
seeds=("$@")
if (( ${#seeds[@]} == 0 )); then seeds=(1017 1019 1021); fi
schedule=()
case "$regime" in
    RB_small) schedule=(--refine-ms 185 --refine-first);;
    RB_large) schedule=(--refine-ms 175);;
    ER_small) ;;
    *) echo "Unknown graph family and size: $regime" >&2; exit 2;;
esac
stage=${BUILD_DIR:-tmp/priority_200ms_build}
mkdir -p "$stage"
for name in priority_search mis_clique_refine; do
    src=models/$name.cpp
    lib="$stage/$name.so"
    if [[ ! -f "$lib" || "$src" -nt "$lib" ]]; then
        ${CXX:-g++} -std=c++17 -O3 -march=native -fPIC -shared "$src" -o "$lib.$$"
        mv "$lib.$$" "$lib"
    fi
done
ranking=()
checkpoint=${CHECKPOINT:-outputs/v11/scale_priority_20261121/training/best.pt}
case "$arm" in
    learned) ranking=(--checkpoint "$checkpoint" --degree-prior --learned-mix 0.25);;
    learned:*) ranking=(--checkpoint "$checkpoint" --degree-prior --learned-mix "${arm#learned:}");;
    neural) ranking=(--checkpoint "$checkpoint" --neural-only);;
    degree) ranking=(--degree-prior);;
    neutral) ;;
    *) echo "Unknown priority arm: $arm" >&2; exit 2;;
esac
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
exec "${PYTHON:-$HOME/envs/rlnn/bin/python}" -u models/eval_priority_search.py \
    --dataset "$dataset" --output "$output" --seeds "${seeds[@]}" --budget-ms 200 --reserve-ms 3 \
    --library "$stage/priority_search.so" --refiner "$stage/mis_clique_refine.so" \
    --weight 0.7 --plateau 50 --perturb 4 "${ranking[@]}" "${schedule[@]}"
