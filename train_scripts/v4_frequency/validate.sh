#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
gpu="${1:?pass GPU 0 or 1}"
export CUDA_VISIBLE_DEVICES="$gpu"
while [[ ! -f "$OUT/queue/screen.finished.txt" ]]; do sleep 20; done
"$PYTHON" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out=Path(sys.argv[1])
for arm in ('none','v4a_mlp','v4a_message','v4a'):
    for seed in range(3):
        p=out/f'{arm}_seed{seed}'/'run.json'
        d=json.loads(p.read_text())
        if d['status'] != 'completed': raise SystemExit(f'Training incomplete: {p}')
PY

logged() {
  local name="$1" rc=0
  shift
  "$@" > "$OUT/queue/$name.log" 2>&1 || rc=$?
  printf '%s\n' "$rc" > "$OUT/queue/$name.exit_code.txt"
  return "$rc"
}

index=0
for precision in no fp16; do
  for label in baseline tuned v4a_mlp v4a_message v4a; do
    for seed in 0 1 2; do
      if ((index % 2 == gpu)); then
        arm="$label"
        train_run="${label}_seed${seed}"
        checkpoint="$OUT/$train_run/checkpoints"
        if [[ "$label" == baseline ]]; then arm=none; checkpoint="$(base_checkpoint "$seed")"; fi
        if [[ "$label" == tuned ]]; then arm=none; train_run="none_seed${seed}"; checkpoint="$OUT/$train_run/checkpoints"; fi
        name="val_${label}_${precision}_seed${seed}"
        SEED="$seed" LABEL="$label" CHECKPOINT="$checkpoint" TRAIN_RUN="$train_run" \
          RUN_NAME="$name" PRECISION="$precision" SPLIT=validation \
          logged "$name" bash train_scripts/v4_frequency/eval.sh "$arm"
      fi
      index=$((index + 1))
    done
  done
done
date -Is > "$OUT/queue/validation_fixed_gpu${gpu}.done.txt"

for label in baseline tuned v4a_mlp v4a_message v4a; do
  for seed in 0 1 2; do
    if ((index % 2 == gpu)); then
      arm="$label"
      checkpoint="$OUT/${label}_seed${seed}/checkpoints"
      if [[ "$label" == baseline ]]; then arm=none; checkpoint="$(base_checkpoint "$seed")"; fi
      if [[ "$label" == tuned ]]; then arm=none; checkpoint="$OUT/none_seed${seed}/checkpoints"; fi
      name="budget_${label}_pave_seed${seed}"
      SEED="$seed" LABEL="$label" CHECKPOINT="$checkpoint" RUN_NAME="$name" \
        PRECISION=fp16 SPLIT=validation POST=pave \
        logged "$name" bash train_scripts/v4_frequency/eval_budget.sh "$arm"
    fi
    index=$((index + 1))
  done
done
for post in none rule23; do
  for seed in 0 1 2; do
    if ((index % 2 == gpu)); then
      name="budget_baseline_${post}_extended_seed${seed}"
      SEED="$seed" LABEL=baseline CHECKPOINT="$(base_checkpoint "$seed")" RUN_NAME="$name" \
        PRECISION=fp16 SPLIT=validation POST="$post" \
        logged "$name" bash train_scripts/v4_frequency/eval_budget.sh none --extended-search --post-reserve-s 0.02
    fi
    index=$((index + 1))
  done
done
date -Is > "$OUT/queue/validation_gpu${gpu}.done.txt"
