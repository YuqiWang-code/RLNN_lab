#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
seed="${SEED:-0}"
model_seed="${MODEL_SEED:-0}"
export PYTHONPATH="$ROOT/models${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -u "$ROOT/models/archived_frequency/probe_v4_utility.py" \
  --teacher-checkpoint "${TEACHER_CHECKPOINT:-$(base_checkpoint 0)/best.pt}" \
  --validation-path "$DATA/MIS_tensors/RB_small_val" \
  --result-dir "$OUT/${RUN_NAME:-utility_seed${seed}}" --seed "$seed" \
  --graphs "${GRAPHS:-20}" --nodes "${NODES:-16}" --horizon "${HORIZON:-20}" \
  --model mlp v4a_mlp "${MLP_CHECKPOINT:-$OUT/v4a_mlp_seed${model_seed}/checkpoints/best.pt}" \
  --model message v4a_message "${MESSAGE_CHECKPOINT:-$OUT/v4a_message_seed${model_seed}/checkpoints/best.pt}" \
  --model frequency v4a "${FREQUENCY_CHECKPOINT:-$OUT/v4a_seed${model_seed}/checkpoints/best.pt}" "$@"
