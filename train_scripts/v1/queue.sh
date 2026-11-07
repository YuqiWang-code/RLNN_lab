#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WANDB_MODE=disabled
out=outputs/v1
mkdir -p "$out/queue"

completed() {
  test -f "$out/$1/run.json" && "$PYTHON" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["status"] != "completed")' "$out/$1/run.json"
}
task() {
  local name=$1 status
  shift
  completed "$name" && return
  printf '%s\t%s\tstart\n' "$(date -Is)" "$name"
  "$@" > "$out/queue/$name.log" 2>&1
  status=$?
  printf '%s\n' "$status" > "$out/queue/$name.exit_code.txt"
  printf '%s\t%s\texit=%s\n' "$(date -Is)" "$name" "$status"
  return "$status"
}
wait_existing() {
  local name=$1 pid status
  if [[ -f "$out/queue/$name.pid" ]]; then
    pid=$(cat "$out/queue/$name.pid")
    while status=$(ps -p "$pid" -o stat=) && [[ -n "$status" && "$status" != Z* ]]; do sleep 45; done
  fi
}
train_frequency() {
  local arm=$1 seed=$2 name
  name="freq_${arm}_seed${seed}"
  wait_existing "$name"
  RUN_NAME="$name" SEED="$seed" task "$name" bash train_scripts/v1/train.sh --frequency "$arm"
}
base_checkpoint() {
  if [[ "$1" == 0 ]]; then
    echo outputs/v000_rlnn_baseline/rb_train_author_b64_g8_gpu1_seed0/checkpoints
  else
    completed "rlnn_seed$1" || return 1
    echo "$out/rlnn_seed$1/checkpoints"
  fi
}
policy() {
  local name=$1 label=$2 seed=$3 checkpoint=$4
  shift 4
  test -f "$checkpoint/best.pt" || return 1
  task "$name" bash train_scripts/v1/eval.sh --label "$label" --result_dir "$out/$name" \
    --save_dir "$checkpoint" --seed 1017 --train_seed "$seed" "$@"
}
head() {
  local name=$1 seed=$2 checkpoint=$3 frequency=$4
  test -f "$checkpoint/best.pt" || return 1
  task "$name" "$PYTHON" -u models/train_flip_value.py --result_dir "$out/$name" \
    --save_dir "$checkpoint" --frequency "$frequency" --seed "$seed" --train_samples 500
}
baseline_arms() {
  local seed=$1 base
  base=$(base_checkpoint "$seed") || return 1
  policy "eval_rlnn_seed$seed" RLNN "$seed" "$base"
  policy "eval_archive_seed$seed" "RLNN + Archive" "$seed" "$base" --archive quality
  policy "eval_archive_energy_seed$seed" "Archive 能量选样" "$seed" "$base" --archive energy
  policy "eval_value_energy_seed$seed" "RLNN + 一步能量" "$seed" "$base" --value_energy
  if head "value_rlnn_seed$seed" "$seed" "$base" none; then
    policy "eval_value_seed$seed" "RLNN + 成对收益" "$seed" "$base" \
      --value_checkpoint "$out/value_rlnn_seed$seed/checkpoints/best.pt"
  fi
}
frequency_arms() {
  local seed=$1 name
  for arm in dynamic stats; do
    name="freq_${arm}_seed${seed}"
    completed "$name" && policy "eval_$name" "分频 $arm" "$seed" "$out/$name/checkpoints" --frequency "$arm"
  done
  name="freq_dynamic_seed$seed"
  if completed "$name" && head "value_dynamic_seed$seed" "$seed" "$out/$name/checkpoints" dynamic; then
    policy "eval_combined_seed$seed" "分频 + Archive + 成对收益" "$seed" "$out/$name/checkpoints" \
      --frequency dynamic --archive quality --value_checkpoint "$out/value_dynamic_seed$seed/checkpoints/best.pt"
  fi
}
classical_gpu() {
  task rlsa bash train_scripts/v1/eval.sh --label RLSA --method RLSA --result_dir "$out/rlsa" \
    --num_t 300 --num_k 200 --num_d 5 --seed 1017
  task isco bash train_scripts/v1/eval.sh --label "iSCO（固定半径）" --method iSCO --result_dir "$out/isco" \
    --num_t 300 --num_k 200 --num_d 5 --tau0 1 --seed 1017
}
reference_training() {
  local method=$1 name=$2 entry=$3
  if [[ "$method" == GCON ]]; then
    task "$name" "$PYTHON" -u "$entry" --label "$method" --result_dir "$out/$name" || return
  else
    task "$name" "$PYTHON" -u "$entry" --label "$method" --method "$method" --do_train --result_dir "$out/$name" || return
  fi
}
reference_eval() {
  local method=$1 name=$2 entry=$3
  completed "$name" || return 1
  if [[ "$method" == GCON ]]; then
    task "eval_$name" "$PYTHON" -u "$entry" --label "$method" --result_dir "$out/eval_$name" \
      --checkpoint "$out/$name/checkpoints/best.pt" --seed 1017
  else
    task "eval_$name" "$PYTHON" -u "$entry" --label "$method" --method "$method" --result_dir "$out/eval_$name" \
      --checkpoint "$out/$name/checkpoints/best.pt" --seed 1017
  fi
}

case "$1" in
  frequency)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    train_frequency dynamic 0 & a=$!
    train_frequency stats 0 & b=$!
    wait "$a"; wait "$b"
    baseline_arms 0
    frequency_arms 0
    base=$(base_checkpoint 0)
    policy archive_diagnostic ArchiveGap 0 "$base" --archive quality --diagnostic --test_samples 32
    for seed in 1 2; do
      train_frequency dynamic "$seed" & a=$!
      train_frequency stats "$seed" & b=$!
      wait "$a"; wait "$b"
      frequency_arms "$seed"
    done
    for arm in low high fixed; do train_frequency "$arm" 0 & done
    wait
    for arm in low high fixed; do
      name="freq_${arm}_seed0"
      completed "$name" && policy "eval_$name" "分频 $arm" 0 "$out/$name/checkpoints" --frequency "$arm"
    done
    policy budget_rlnn RLNN 0 "$base" --time_budget_s 0.5
    policy budget_archive "RLNN + Archive" 0 "$base" --archive quality --time_budget_s 0.5
    policy budget_archive_energy "Archive 能量选样" 0 "$base" --archive energy --time_budget_s 0.5
    policy budget_dynamic "分频 dynamic" 0 "$out/freq_dynamic_seed0/checkpoints" --frequency dynamic --time_budget_s 0.5
    if completed value_rlnn_seed0; then
      policy budget_value "RLNN + 成对收益" 0 "$base" --time_budget_s 0.5 \
        --value_checkpoint "$out/value_rlnn_seed0/checkpoints/best.pt"
    fi
    if completed value_dynamic_seed0; then
      policy budget_combined "分频 + Archive + 成对收益" 0 "$out/freq_dynamic_seed0/checkpoints" \
        --frequency dynamic --archive quality --time_budget_s 0.5 \
        --value_checkpoint "$out/value_dynamic_seed0/checkpoints/best.pt"
    fi
    ;;
  comparison)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
    for seed in 1 2; do
      wait_existing "rlnn_seed$seed"
      if ! completed "rlnn_seed$seed"; then
        SEED="$seed" task "rlnn_seed$seed" bash train_scripts/v1/train.sh
      fi
    done
    for seed in 1 2; do baseline_arms "$seed"; done
    classical_gpu
    gcon=train_scripts/v1/reference_gcon.py
    cam=train_scripts/v1/reference_cam.py
    reference_training GCON gcon_train "$gcon"
    reference_eval GCON gcon_train "$gcon"
    reference_training CAM cam_train "$cam" & a=$!
    reference_training DiffUCO diffuco_train "$cam" & b=$!
    wait "$a"; wait "$b"
    reference_eval CAM cam_train "$cam"
    reference_eval DiffUCO diffuco_train "$cam"
    ;;
  *) printf 'Usage: bash train_scripts/v1/queue.sh frequency|comparison\n' >&2; exit 2 ;;
esac
printf '%s\n' "$(date -Is)" > "$out/queue/$1.done.txt"
