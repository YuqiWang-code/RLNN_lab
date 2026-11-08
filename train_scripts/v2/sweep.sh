#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
stage="${1:?pass train or test}"
gpu="${2:?pass GPU 0 or 1}"
export CUDA_VISIBLE_DEVICES="$gpu"
mkdir -p "$OUT/queue"

if [[ "$gpu" == 0 ]]; then
  arms=(segf:0 segf:2 none:0 none:2 segf_stats:0 segf_static2:0 segf_fixed:0)
  seeds=(0 2)
else
  arms=(segf:1 none:1 segf_mlp:0 segf_static1:0 segf_exchange:0 segf_high:0)
  seeds=(1)
fi

logged() {
  local name="$1" rc=0
  shift
  "$@" > "$OUT/queue/$name.log" 2>&1 || rc=$?
  printf '%s\n' "$rc" > "$OUT/queue/$name.exit_code.txt"
  return "$rc"
}

if [[ "$stage" == train ]]; then
  pids=()
  for experiment in "${arms[@]}"; do
    arm="${experiment%:*}"
    seed="${experiment#*:}"
    run="${arm}_seed${seed}"
    SEED="$seed" RUN_NAME="$run" logged "$run" bash train_scripts/v2/train_frequency.sh "$arm" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if ((failed)); then exit 1; fi
elif [[ "$stage" == test ]]; then
  suite() {
    local name="$1" seed="$2" checkpoint="$3" arm="$4" label="$5"
    shift 5
    logged "$name" bash train_scripts/v2/eval.sh --result_dir "$OUT/$name" --train_seed "$seed" \
      --save_dir "$checkpoint" --frequency "$arm" --label "$label" "$@"
  }
  for seed in "${seeds[@]}"; do
    base="$(base_checkpoint "$seed")"
    tuned="$OUT/none_seed${seed}/checkpoints"
    segf="$OUT/segf_seed${seed}/checkpoints"
    suite "val_base_seed$seed" "$seed" "$base" none RLNN --split validation
    suite "val_tuned_seed$seed" "$seed" "$tuned" none RLNN-tuned --split validation
    suite "val_segf_seed$seed" "$seed" "$segf" segf SEGF --split validation
    suite "test_base_seed$seed" "$seed" "$base" none RLNN --post none rule heuristic random rule23
  done
  for experiment in "${arms[@]}"; do
    arm="${experiment%:*}"
    seed="${experiment#*:}"
    run="${arm}_seed${seed}"
    case "$arm" in
      none) suite "test_$run" "$seed" "$OUT/$run/checkpoints" "$arm" RLNN-tuned ;;
      segf) suite "test_$run" "$seed" "$OUT/$run/checkpoints" "$arm" SEGF --post none rule23 ;;
      *) suite "test_$run" "$seed" "$OUT/$run/checkpoints" "$arm" "$arm" ;;
    esac
  done
  segf_seeds=("${seeds[@]}")
  if [[ "$gpu" == 0 ]]; then
    segf_seeds=(0)
  else
    segf_seeds=(1 2)
  fi
  for seed in "${seeds[@]}"; do
    base="$(base_checkpoint "$seed")"
    suite "budget_base_seed$seed" "$seed" "$base" none RLNN --time_budget_s 0.5
    suite "budget_base_rule23_seed$seed" "$seed" "$base" none RLNN --post rule23 --time_budget_s 0.5
  done
  for seed in "${segf_seeds[@]}"; do
    segf="$OUT/segf_seed${seed}/checkpoints"
    suite "budget_segf_seed$seed" "$seed" "$segf" segf SEGF --time_budget_s 0.5
    suite "budget_segf_rule23_seed$seed" "$seed" "$segf" segf SEGF --post rule23 --time_budget_s 0.5
  done
else
  printf 'Unknown stage %s\n' "$stage" >&2
  exit 1
fi
date -Is > "$OUT/queue/${stage}_gpu${gpu}.done.txt"
