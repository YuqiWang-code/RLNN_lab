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
fp32=(); fp16=()
for seed in 0 1 2; do
  name="val_fp32_seed$seed"
  run "$name" 0 models/archived_frequency/diagnose_v4_branch_replay.py --seed "$seed" --split validation --graphs 100 --precision fp32 --native-repeats 2 --result-dir "$OUT/p2_branch/$name" &
  fp32+=("$!")
  name="val_fp16_seed$seed"
  run "$name" 1 models/archived_frequency/diagnose_v4_branch_replay.py --seed "$seed" --split validation --graphs 100 --precision fp16 --native-repeats 2 --result-dir "$OUT/p2_branch/$name" &
  fp16+=("$!")
done
code=0
for pid in "${fp32[@]}"; do wait "$pid" || code=1; done
if [[ "$code" != 0 ]]; then exit "$code"; fi
date -Is > "$OUT/queue/branch_fp32.done.txt"
for pid in "${fp16[@]}"; do wait "$pid" || code=1; done
if [[ "$code" != 0 ]]; then exit "$code"; fi
date -Is > "$OUT/queue/branch_confirm.done.txt"
