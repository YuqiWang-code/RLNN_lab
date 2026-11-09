#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
cd "$ROOT"
while [[ ! -f "$OUT/queue/screen.done.txt" || ! -f "$OUT/queue/preliminary.done.txt" ]]; do sleep 10; done

fixed_lane() {
  local gpu="$1" precision="$2" index=0 arm seed name
  local posts=(none)
  if [[ "$precision" == fp16 ]]; then posts+=(pave); fi
  for arm in baseline none v5a_mlp v5a_message v5a analytic; do
    for seed in 0 1 2; do
      if (( index % 2 == gpu )); then
        name="eval_fixed_${arm}_seed${seed}_$precision"
        CUDA_VISIBLE_DEVICES="$gpu" SEED="$seed" PRECISION="$precision" bash train_scripts/v5_frequency/eval.sh "$arm" \
          --post "${posts[@]}" > "$OUT/queue/$name.log" 2>&1
      fi
      index=$((index+1))
    done
  done
}

budget_lane() {
  local gpu="$1" index=0 budget method arm seed post name
  for budget in 0.5 1.0; do
    for method in baseline baseline_extended baseline_rule23 v5a_mlp v5a_message v5a; do
      for seed in 0 1 2; do
        if (( index % 2 == gpu )); then
          arm="$method"; post=pave
          local extension=()
          if [[ "$method" == baseline_* ]]; then arm=baseline; extension=(--extended-search); fi
          if [[ "$method" == baseline_rule23 ]]; then post=rule23; fi
          name="eval_budget_${budget}_${method}_seed${seed}"
          CUDA_VISIBLE_DEVICES="$gpu" SEED="$seed" BUDGET="$budget" POST="$post" LABEL="$method" RUN_NAME="$name" \
            bash train_scripts/v5_frequency/eval_budget.sh "$arm" "${extension[@]}" > "$OUT/queue/$name.log" 2>&1
        fi
        index=$((index+1))
      done
    done
  done
}

join() {
  local status=0
  for pid in "$@"; do wait "$pid" || status=1; done
  return "$status"
}

for precision in no fp16; do
  fixed_lane 0 "$precision" & first=$!
  fixed_lane 1 "$precision" & second=$!
  join "$first" "$second"
  "$PYTHON" train_scripts/v5_frequency/decide.py
done
date -Is > "$OUT/queue/fixed.done.txt"
budget_lane 0 & first=$!
budget_lane 1 & second=$!
join "$first" "$second"
"$PYTHON" train_scripts/v5_frequency/decide.py
date -Is > "$OUT/queue/budget.done.txt"
