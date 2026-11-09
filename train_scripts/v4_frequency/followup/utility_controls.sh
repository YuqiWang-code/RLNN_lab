#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
OUT="$ROOT/outputs/v4_frequency/followup"
pids=()
run() {
  local name="$1" gpu="$2"; shift 2
  local code=0
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$@" > "$OUT/queue/$name.log" 2>&1 || code=$?
  echo "$code" > "$OUT/queue/$name.exit_code.txt"
  return "$code"
}
join() { local code=0; for pid in "${pids[@]}"; do wait "$pid" || code=1; done; pids=(); return "$code"; }
for seed in 0 1 2; do
  for horizon in 10 40; do
    if [[ "$horizon" == 10 ]]; then gpu=0; else gpu=1; fi
    name="reliability_h${horizon}_seed$seed"
    run "$name" "$gpu" models/probe_v4_utility_followup.py --seed "$seed" --split train --graphs 10 --nodes 16 --replicates 16 --sampling uniform --phases 25 75 --horizon "$horizon" --result-dir "$OUT/p3_utility/$name" &
    pids+=("$!")
  done
done
join
for seed in 0 1 2; do
  for source in F4 MSG; do
    if [[ "$source" == F4 ]]; then gpu=0; else gpu=1; fi
    name="own_${source}_seed$seed"
    run "$name" "$gpu" models/probe_v4_utility_followup.py --seed "$seed" --split train --graphs 20 --nodes 24 --replicates 8 --sampling stratified --phases 25 50 100 --horizon 20 --trajectory-model "$source" --result-dir "$OUT/p3_utility/$name" &
    pids+=("$!")
  done
done
join
date -Is > "$OUT/queue/utility_controls.done.txt"
