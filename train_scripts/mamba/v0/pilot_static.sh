#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
export REF_SEED=0 WIDTH=32 SAMPLES=50
pilot="$OUT/static_prior_pilot"
stage="${1:-full}"
export EPOCHS=1

train() {
  local arm="$1" gpu="$2"
  CUDA_VISIBLE_DEVICES="$gpu" RUN_NAME="static_prior_pilot/train_$arm" \
    bash "$SCRIPTS/train.sh" "v7_$arm" --train_samples 256 --valid_samples "$SAMPLES"
}

if [[ "$stage" == full ]]; then
  mkdir "$pilot"
  train mlp 0 >"$pilot/train_mlp.log" 2>&1 & mlp_pid=$!
  train message 1 >"$pilot/train_message.log" 2>&1 & msg_pid=$!
  wait "$mlp_pid"
  wait "$msg_pid"

  "$PYTHON" - "$pilot" "$OUT/../v2/none_seed0/checkpoints/best.pt" <<'PY'
import json
from pathlib import Path
import sys
import torch

pilot, source = map(Path, sys.argv[1:])
backbone = torch.load(source, map_location="cpu", weights_only=True)
matched = []
for arm in ("mlp", "message"):
    folder = pilot / f"train_{arm}"
    data = json.loads((folder / "run.json").read_text())
    assert data["status"] == "completed" and data["updates"] == 512
    matched.append(data["teacher_state_sha256"])
    weights = torch.load(folder / "checkpoints/best.pt", map_location="cpu", weights_only=True)
    assert all(torch.equal(weights[name], value) for name, value in backbone.items()), arm
assert len(matched[0]) == 20 and matched[0] == matched[1]
print("512 updates per arm, 20 paired teacher traces and identical frozen backbones", flush=True)
PY
elif [[ "$stage" != repeat-budgets || ! -f "$pilot/completed.txt" ]]; then
  echo "Pass full or repeat-budgets after the pilot finishes" >&2
  exit 2
fi

measure() {
  local protocol="$1" arm="$2" gpu="$3" budget="${4:-}"
  local mode="$arm" checkpoint="$OUT/../v2/none_seed0/checkpoints" script=eval.sh
  if [[ "$arm" != baseline ]]; then
    mode="v7_$arm"
    checkpoint="$pilot/train_$arm/checkpoints"
  fi
  if [[ -n "$budget" ]]; then script=eval_budget.sh; fi
  CUDA_VISIBLE_DEVICES="$gpu" RUN_NAME="static_prior_pilot/${protocol}_$arm" \
    CHECKPOINT_DIR="$checkpoint" BUDGET="$budget" bash "$SCRIPTS/$script" "$mode" --cuda-graph \
    >"$pilot/${protocol}_$arm.log" 2>&1
}

first_lane() {
  if [[ "$stage" == full ]]; then
    for arm in baseline mlp message; do measure fixed "$arm" 0; done
    for arm in message baseline mlp; do measure budget05 "$arm" 0 0.5; done
  else
    for arm in mlp baseline message; do measure repeat05 "$arm" 0 0.5; done
  fi
}

second_lane() {
  if [[ "$stage" == full ]]; then
    for arm in mlp message baseline; do measure budget10 "$arm" 1 1.0; done
  else
    for arm in baseline message mlp; do measure repeat10 "$arm" 1 1.0; done
  fi
}

first_lane & first=$!
second_lane & second=$!
wait "$first"
wait "$second"
if [[ "$stage" == full ]]; then
  date -Is >"$pilot/completed.txt"
else
  date -Is >"$pilot/repeat_completed.txt"
fi
