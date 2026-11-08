#!/usr/bin/env bash
set -euo pipefail
train_data="${1:?pass TRAIN_DATA VALID_DATA MODE SEED NAME GPU}"
valid_data="${2:?pass VALID_DATA}"
mode="${3:?pass stats, message, or frequency}"
seed="${4:?pass SEED}"
name="${5:?pass NAME}"
gpu="${6:?pass GPU}"
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
export CUDA_VISIBLE_DEVICES="$gpu"
decision="${DECISION:-$OUT/decision.json}"
"$PYTHON" - "$decision" "$train_data" "$valid_data" <<'PY'
import json
from pathlib import Path
import sys

decision_path, train_data, valid_data = map(Path, sys.argv[1:])
decision = json.loads(decision_path.read_text(encoding="utf-8"))
if decision.get("train") is not True:
    reasons = "; ".join(decision.get("reasons", []))
    raise SystemExit(f"Formal training is not admitted by {decision_path}: {reasons}")
for name, actual in (("training", train_data), ("validation", valid_data)):
    expected = Path(decision[name]["states_path"])
    if actual.resolve() != expected.resolve():
        raise SystemExit(f"{name} data does not match the admitted labels: {actual} != {expected}")
print(f"Training admitted: {decision['variant']} ({decision_path})", flush=True)
PY
exec "$PYTHON" -u "$ROOT/models/train_exchange_value_v3.py" \
  --train-data "$train_data" --valid-data "$valid_data" --mode "$mode" --seed "$seed" \
  --result-dir "$OUT/$name" --device cuda:0 --epochs 30 --batch-size 32 --max-pairs 64
