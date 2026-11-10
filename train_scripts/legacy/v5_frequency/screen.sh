#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
mkdir -p "$OUT/queue"
pids=()
i=0
for arm in none v5a_mlp v5a_message v5a; do
  for seed in 0 1 2; do
    gpu=$((i % 2))
    cpu=$((gpu * 24 + i / 2 * 4))
    run="${arm}_seed${seed}"
    (
      code=0
      CUDA_VISIBLE_DEVICES="$gpu" SEED="$seed" taskset -c "$cpu-$((cpu+3))" bash train_scripts/legacy/v5_frequency/train.sh "$arm" || code=$?
      printf '%s\n' "$code" > "$OUT/queue/$run.exit_code.txt"
      exit "$code"
    ) > "$OUT/queue/$run.log" 2>&1 &
    pids+=("$!")
    i=$((i+1))
  done
done
code=0
for pid in "${pids[@]}"; do wait "$pid" || code=1; done
if (( code == 0 )); then date -Is > "$OUT/queue/screen.done.txt"; fi
exit "$code"
