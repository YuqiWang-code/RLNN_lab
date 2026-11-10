#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
mkdir -p "$OUT/queue"
arms=(none v4a_mlp v4a_message v4a)
pids=()
index=0
for arm in "${arms[@]}"; do
  for seed in 0 1 2; do
    gpu=$((index % 2))
    slot=$((index / 2))
    cpu_first=$((gpu * 24 + slot * 4))
    cpu_last=$((cpu_first + 3))
    run="${arm}_seed${seed}"
    (
      rc=0
      CUDA_VISIBLE_DEVICES="$gpu" SEED="$seed" taskset -c "$cpu_first-$cpu_last" \
        bash "$ROOT/train_scripts/legacy/v4_frequency/train.sh" "$arm" > "$OUT/queue/$run.log" 2>&1 || rc=$?
      printf '%s\n' "$rc" > "$OUT/queue/$run.exit_code.txt"
      exit "$rc"
    ) &
    pids+=("$!")
    printf '%s GPU%s pid=%s\n' "$run" "$gpu" "$!" >> "$OUT/queue/screen.pids.txt"
    index=$((index + 1))
  done
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
date -Is > "$OUT/queue/screen.finished.txt"
exit "$failed"
