#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WANDB_MODE=disabled
out=outputs/v1
mkdir -p "$out/queue"

start() {
  local gpu=$1 name=$2 reserve=$3 parent=$4 free pid
  shift 4
  if [[ -f "$out/queue/$name.pid" ]]; then
    pid=$(cat "$out/queue/$name.pid")
    if ps -p "$pid" -o stat= | grep -q '^[^Z]'; then return; fi
  fi
  if [[ -d "$out/$name" ]]; then return; fi
  if [[ -z "$parent" ]]; then
    free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
    if (( free < reserve )); then
      printf '%s stays queued on GPU %s (%s MiB free)\n' "$name" "$gpu" "$free"
      return
    fi
  elif [[ ! -f "$out/queue/$parent.pid" && ! -f "$out/$parent/run.json" ]]; then
    return
  fi
  printf '%s\n' "$gpu" > "$out/queue/$name.gpu.txt"
  CUDA_VISIBLE_DEVICES="$gpu" nohup bash -c '
    name=$1; parent=$2; shift 2
    if [[ -n "$parent" ]]; then
      parent_pid=outputs/v1/queue/$parent.pid
      if [[ -f "$parent_pid" ]]; then
        pid=$(cat "$parent_pid")
        while ps -p "$pid" -o stat= | grep -q "^[^Z]"; do sleep 30; done
      fi
      "$PYTHON" -c "import json, sys; sys.exit(json.load(open(sys.argv[1]))[\"status\"] != \"completed\")" \
        "outputs/v1/$parent/run.json" || exit 1
    fi
    "$@"
    status=$?
    printf "%s\n" "$status" > "outputs/v1/queue/$name.exit_code.txt"
    exit "$status"
  ' bash "$name" "$parent" "$@" > "$out/queue/$name.log" 2>&1 </dev/null &
  printf '%s\n' "$!" > "$out/queue/$name.pid"
  printf '%s\tGPU %s\tPID %s\t%s\n' "$name" "$gpu" "$!" "${parent:+after $parent}"
  sleep 3
}

frequency() {
  local gpu=$1 arm=$2 seed=$3 name
  name="freq_${arm}_seed${seed}"
  start "$gpu" "$name" 6000 '' env RUN_NAME="$name" SEED="$seed" \
    bash train_scripts/v1/train.sh --frequency "$arm" --num_workers 2
}

cam=train_scripts/v1/reference_cam.py
start 0 cam_train 25000 '' "$PYTHON" -u "$cam" --label CAM --method CAM --do_train --result_dir "$out/cam_train"
start 1 diffuco_train 12500 '' "$PYTHON" -u "$cam" --label DiffUCO --method DiffUCO --do_train --result_dir "$out/diffuco_train"
frequency 0 dynamic 1
frequency 1 dynamic 2
frequency 0 stats 2
frequency 1 stats 1
frequency 1 low 0
frequency 1 high 0
frequency 1 fixed 0
start 1 value_rlnn_seed0 5000 '' "$PYTHON" -u models/train_flip_value.py \
  --result_dir "$out/value_rlnn_seed0" \
  --save_dir outputs/v000_rlnn_baseline/rb_train_author_b64_g8_gpu1_seed0/checkpoints \
  --frequency none --seed 0 --train_samples 500

head_after() {
  local gpu=$1 name=$2 parent=$3 arm=$4 seed=$5
  start "$gpu" "$name" 0 "$parent" "$PYTHON" -u models/train_flip_value.py \
    --result_dir "$out/$name" --save_dir "$out/$parent/checkpoints" \
    --frequency "$arm" --seed "$seed" --train_samples 500
}

for seed in 1 2; do head_after 1 "value_rlnn_seed$seed" "rlnn_seed$seed" none "$seed"; done
for seed in 0 1; do head_after 0 "value_dynamic_seed$seed" "freq_dynamic_seed$seed" dynamic "$seed"; done
head_after 1 value_dynamic_seed2 freq_dynamic_seed2 dynamic 2
