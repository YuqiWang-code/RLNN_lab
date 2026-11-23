#!/usr/bin/env bash
set -euo pipefail

root=${RLNN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$root"
phase=${1:?usage: eval_online_mis.sh validation|test SPLIT CPU}
split=${2:?missing graph distribution}
core=${3:?missing logical CPU on a distinct physical core}
case "$split" in RB_small|RB_large|ER_small|ER_large) ;; *) exit 2 ;; esac
python=${PYTHON:-"$HOME/envs/rlnn/bin/python"}
entry=${REFERENCE_ENTRY:-models/eval_reference_deadline.py}
base=${ONLINE_MIS_OUTPUT:-outputs/v15/priority_mis_20261121/online_mis_tuned}
val="$base/validation/$split"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$root/models${PYTHONPATH:+:$PYTHONPATH}"
options=(kamis --kamis tmp/kamis_build/online_mis)

if [[ "$phase" == validation ]]; then
    dataset="models/data/scale_priority_20261121/$split/validation"
    if [[ "$split" == ER_large && ! -f "$dataset/tensor_manifest.json" ]]; then
        taskset -c "$core" "$python" train_scripts/v15/prepare_er_large_validation.py "$dataset"
    fi
    mkdir -p "$val"
    for limit in 120 150 180; do
        taskset -c "$core" "$python" -u "$entry" "${options[@]}" --dataset "$dataset" \
            --graphs 16 --seeds 517 --kamis-internal-ms "$limit" --output "$val/ms$limit"
    done
    "$python" - "$val" "$core" <<'PY'
import csv
import json
from pathlib import Path
from statistics import fmean
import sys

root = Path(sys.argv[1])
stats = []
for limit in (120, 150, 180):
    with (root / f"ms{limit}/per_graph.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    expected = {(f"idx_{i}", "517") for i in range(16)}
    if len(rows) != 16 or {(r["graph_id"], r["seed"]) for r in rows} != expected:
        raise ValueError(f"Incomplete validation sample at {limit} ms")
    stats.append(dict(internal_ms=limit, mean_size=fmean(int(r["final_size"]) for r in rows),
                      mean_wall_ms=fmean(float(r["wall_ms"]) for r in rows),
                      on_time=sum(bool(r["final_ready_ms"]) for r in rows), runs=len(rows)))
selected = max(stats, key=lambda r: (r["mean_size"], r["internal_ms"]))
config = dict(method="OnlineMIS", host_budget_ms=200, cpu=int(sys.argv[2]), validation_seed=517,
              rule="maximum mean deadline-scored size, ties choose longer internal limit",
              selected_internal_ms=selected["internal_ms"], validation=stats)
(root / "selection.json").write_text(json.dumps(config, indent=2) + "\n")
print(root, selected, flush=True)
PY
elif [[ "$phase" == test ]]; then
    limit=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_internal_ms"])' "$val/selection.json")
    case "$split" in ER_small) n=128 ;; ER_large) n=16 ;; *) n=1000 ;; esac
    dataset="models/data/MIS_tensors/${split}_author_test"
    exec taskset -c "$core" "$python" -u "$entry" "${options[@]}" --dataset "$dataset" \
        --graphs "$n" --seeds 1017 1019 1021 --kamis-internal-ms "$limit" --output "$base/test/$split"
else
    exit 2
fi
