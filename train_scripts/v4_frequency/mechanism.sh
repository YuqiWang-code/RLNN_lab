#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
while [[ ! -f "$OUT/queue/validation_gpu0.done.txt" || ! -f "$OUT/queue/validation_gpu1.done.txt" ]]; do sleep 20; done

logged() {
  local name="$1" rc=0
  shift
  "$@" > "$OUT/queue/$name.log" 2>&1 || rc=$?
  printf '%s\n' "$rc" > "$OUT/queue/$name.exit_code.txt"
  return "$rc"
}

pids=()
index=0
for arm in none v4a_mlp v4a_message v4a; do
  for seed in 0 1 2; do
    gpu=$((index % 2))
    name="diagnostic_${arm}_seed${seed}"
    CUDA_VISIBLE_DEVICES="$gpu" SEED="$seed" SAMPLES=20 SPECTRAL_GRAPHS=20 RUN_NAME="$name" \
      logged "$name" bash train_scripts/v4_frequency/diagnose.sh "$arm" &
    pids+=("$!")
    index=$((index + 1))
  done
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if ((failed)); then exit 1; fi
pids=()
for seed in 0 1 2; do
  name="utility_seed${seed}"
  CUDA_VISIBLE_DEVICES="$((seed % 2))" SEED="$seed" MODEL_SEED="$seed" \
    TEACHER_CHECKPOINT="$(base_checkpoint "$seed")/best.pt" RUN_NAME="$name" \
    logged "$name" bash train_scripts/v4_frequency/probe_utility.sh \
      --model tuned none "$OUT/none_seed${seed}/checkpoints/best.pt" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if ((failed)); then exit 1; fi
date -Is > "$OUT/queue/mechanism.done.txt"

interventions() {
  local gpu="$1" index=0 name seed intervention
  for intervention in low high gate; do
    for seed in 0 1 2; do
      if ((index % 2 == gpu)); then
        name="ablation_disable_${intervention}_seed${seed}"
        CUDA_VISIBLE_DEVICES="$gpu" SEED="$seed" LABEL=v4a RUN_NAME="$name" SPLIT=validation \
          logged "$name" bash train_scripts/v4_frequency/eval.sh v4a "--disable-$intervention"
      fi
      index=$((index + 1))
    done
  done
}
interventions 0 &
first=$!
interventions 1 &
second=$!
wait "$first"
wait "$second"
date -Is > "$OUT/queue/interventions.done.txt"
