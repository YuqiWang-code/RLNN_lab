#!/usr/bin/env bash
set -euo pipefail
root=${RLNN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$root"
regime=${1:?usage: anytime.sh REGIME ARM DATASET OUTPUT BUDGET_MS [SEED ...]}
arm=${2:?expected online_mis, learned or degree}
dataset=${3:?expected tensor manifest directory}
out=${4:?expected new output directory}
budget=${5:?expected 50, 100, 200 or 500 milliseconds}
shift 5
case "$budget" in 50|100|200|500) ;; *) exit 2;; esac
seeds=("$@")
if (( ${#seeds[@]} == 0 )); then seeds=(1017 1019 1021); fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python=${PYTHON:-$HOME/envs/rlnn/bin/python}
stage=${BUILD_DIR:-tmp/v16_anytime_build}
mkdir -p "$stage"
args=(--budget-ms "$budget" --reserve-ms 3 --seeds "${seeds[@]}" --dataset "$dataset" --output "$out")
if [[ "$arm" == online_mis ]]; then
    exec "$python" -u models/eval_online_mis_resident.py "${args[@]}" --library "$stage/online_mis.so"
fi
schedule=()
case "$regime" in
    RB_small) schedule=(--refine-ms "$((185*budget/200))" --refine-first);;
    RB_large) schedule=(--refine-ms "$((175*budget/200))");;
    ER_small) ;;
    *) echo "Unknown schedule: $regime" >&2; exit 2;;
esac
ranking=(--degree-prior)
case "$arm" in
    learned) ranking+=(--checkpoint "${CHECKPOINT:-outputs/v11/scale_priority_20261121/training/best.pt}" --learned-mix 0.25);;
    degree) ;;
    *) echo "Unknown method: $arm" >&2; exit 2;;
esac
for name in priority_search mis_clique_refine; do
    src=models/$name.cpp
    lib="$stage/$name.so"
    if [[ ! -f "$lib" || "$src" -nt "$lib" ]]; then
        ${CXX:-g++} -std=c++17 -O3 -march=native -fPIC -shared "$src" -o "$lib.$$"
        mv "$lib.$$" "$lib"
    fi
done
exec "$python" -u models/eval_priority_search.py "${args[@]}" --library "$stage/priority_search.so" \
    --refiner "$stage/mis_clique_refine.so" --weight 0.7 --plateau 50 --perturb 4 "${ranking[@]}" "${schedule[@]}"
