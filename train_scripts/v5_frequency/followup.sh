#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
while [[ ! -f "$OUT/queue/budget.done.txt" ]]; do sleep 10; done

CUDA_VISIBLE_DEVICES=0 bash train_scripts/v5_frequency/profile.sh > "$OUT/queue/profile.log" 2>&1 &
profile=$!
(
  for seed in 0 1 2; do
    CUDA_VISIBLE_DEVICES=1 SEED="$seed" bash train_scripts/v5_frequency/diagnose.sh --utility
    CUDA_VISIBLE_DEVICES=1 SEED="$seed" SPLIT=validation SAMPLES=100 bash train_scripts/v5_frequency/diagnose.sh
  done
  CUDA_VISIBLE_DEVICES=1 SPLIT=validation SAMPLES=20 PRECISION=fp16 \
    bash train_scripts/v5_frequency/diagnose.sh --repeat-graphs 20
) > "$OUT/queue/diagnose.log" 2>&1 &
diagnostic=$!
wait "$profile"
wait "$diagnostic"

quality="$("$PYTHON" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["quality_pass"]))' "$OUT/analysis/decision.json")"
if [[ "$quality" == 0 ]]; then
  pids=()
  i=0
  for arm in v5a_conflict v5a_mlp_conflict v5a_message_conflict; do
    CUDA_VISIBLE_DEVICES="$((i%2))" SEED=0 bash train_scripts/v5_frequency/train.sh "$arm" > "$OUT/queue/${arm}_seed0.log" 2>&1 &
    pids+=("$!")
    i=$((i+1))
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  i=0
  for arm in v5a_conflict v5a_mlp_conflict v5a_message_conflict; do
    CUDA_VISIBLE_DEVICES="$((i%2))" SEED=0 bash train_scripts/v5_frequency/eval.sh "$arm" --post none \
      > "$OUT/queue/eval_fixed_${arm}_seed0_no.log" 2>&1
    i=$((i+1))
  done
  "$PYTHON" train_scripts/v5_frequency/decide.py
fi
"$PYTHON" - <<'PY'
import hashlib
import json
from pathlib import Path

manifest = json.loads(Path("outputs/v4_frequency/followup/manifest.json").read_text(encoding="utf-8"))
frozen = [row for row in manifest["checkpoints"] if row["model"] == "PAVE"]
assert len(frozen) == 3
for row in frozen:
    assert hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() == row["sha256"], row["path"]
Path("outputs/v5_frequency/preflight/pave_unchanged.json").write_text(
    json.dumps({"status": "passed", "checkpoints": frozen}, indent=2) + "\n", encoding="utf-8")
PY
date -Is > "$OUT/queue/followup.done.txt"
