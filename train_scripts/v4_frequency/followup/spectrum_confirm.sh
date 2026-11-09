#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
OUT="$ROOT/outputs/v4_frequency/followup"
run() {
  local name="$1" gpu="$2"; shift 2
  local code=0
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$@" > "$OUT/queue/$name.log" 2>&1 || code=$?
  echo "$code" > "$OUT/queue/$name.exit_code.txt"
  return "$code"
}
for seed in 0 1 2; do
  pids=()
  for source in teacher own; do
    if [[ "$source" == teacher ]]; then gpu=0; else gpu=1; fi
    for model in v4a v4a_message v4a_mlp; do
      name="val_${model}_${source}_seed$seed"
      run "$name" "$gpu" models/diagnose_v4_spectrum.py --seed "$seed" --split validation --graphs 100 --models "$model" --trajectory "$source" --result-dir "$OUT/p1_spectrum/$name" &
      pids+=("$!")
    done
  done
  code=0
  for pid in "${pids[@]}"; do wait "$pid" || code=1; done
  if [[ "$code" != 0 ]]; then exit "$code"; fi
done
date -Is > "$OUT/queue/spectrum_confirm.done.txt"
