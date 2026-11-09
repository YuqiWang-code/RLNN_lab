#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
while [[ ! -f "$OUT/queue/interventions.done.txt" ]]; do sleep 20; done

run_card() {
  local gpu="$1" seed name rc
  for seed in 0 1 2; do
    if ((seed % 2 != gpu)); then continue; fi
    name="budget_baseline_pave_extended_seed${seed}"
    rc=0
    CUDA_VISIBLE_DEVICES="$gpu" SEED="$seed" LABEL=baseline \
      CHECKPOINT="$(base_checkpoint "$seed")" RUN_NAME="$name" \
      PRECISION=fp16 SPLIT=validation POST=pave \
      bash train_scripts/v4_frequency/eval_budget.sh none --extended-search \
      > "$OUT/queue/$name.log" 2>&1 || rc=$?
    printf '%s\n' "$rc" > "$OUT/queue/$name.exit_code.txt"
    if ((rc)); then return "$rc"; fi
  done
}
run_card 0 &
first=$!
run_card 1 &
second=$!
wait "$first"
wait "$second"
date -Is > "$OUT/queue/strong_pave.done.txt"
