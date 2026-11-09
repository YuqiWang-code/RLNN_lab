#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
q="$OUT/queue"
for ref in 0 1 2; do
  pilot="$OUT/pilot_full_ref$ref"
  until [[ -f "$pilot/training.csv" ]]; do
    pid="$(cat "$q/pilot_full_ref$ref.pid")"
    if ! ps -p "$pid" -o stat= | grep -q '^[^Z ]'; then
      echo "Pilot $ref exited before completing training" >&2
      exit 1
    fi
    sleep 5
  done
  export REF_SEED="$ref"
  RUN_NAME="pilot_fixed_baseline_ref$ref" bash "$ROOT/train_scripts/v6/eval.sh" baseline --cuda-graph
  CHECKPOINT_DIR="$pilot/checkpoints" RUN_NAME="pilot_fixed_full_ref$ref" \
    bash "$ROOT/train_scripts/v6/eval.sh" v6_full --cuda-graph
done
for precision in fp16 no; do
  "$PYTHON" -u "$ROOT/models/profile_v6_frequency.py" --precision "$precision" --cuda-graph \
    --checkpoint "$OUT/pilot_full_ref0/checkpoints" --output "$OUT/profile_pilot_${precision}" "$@"
done
