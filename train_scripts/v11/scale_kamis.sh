#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
split=${1:?expected RB_large, ER_small or ER_large}
arm=${2:?expected online_mis or redumis}
out=${3:?expected a new output directory}
case "$split" in RB_large|ER_small|ER_large) ;; *) exit 2 ;; esac
case "$arm" in online_mis|redumis) ;; *) exit 2 ;; esac
python=${PYTHON:-"$HOME/envs/rlnn/bin/python"}
dataset="models/data/MIS_tensors/${split}_author_test"
count=$("$python" -c 'import json,sys;print(len(json.load(open(sys.argv[1]))["graphs"]))' "$dataset/tensor_manifest.json")
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$root/models${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" -u "${REFERENCE_ENTRY:-models/eval_reference_deadline.py}" kamis --dataset "$dataset" \
    --graphs "$count" --seeds 1017 1019 1021 --kamis "tmp/kamis_build/$arm" --output "$out"
