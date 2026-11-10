#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
horizon="$($PYTHON - "$OUT/model_decision.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if not d['train_encoders']:
    raise SystemExit('Statistics ranker has not exceeded the strongest validation rule')
print(d['stats'].rsplit('_h',1)[1])
PY
)"
decision="$OUT/decision_h$horizon.json"
if [[ "$horizon" == 2 ]]; then decision="$OUT/decision.json"; fi
pids=()
task=0
for mode in message frequency; do
  for seed in 0 1 2; do
    name="${mode}_h${horizon}_seed${seed}"
    DECISION="$decision" bash train_scripts/legacy/v3/train.sh "$OUT/labels_train_h$horizon/states.pt" \
      "$OUT/labels_validation_h$horizon/states.pt" "$mode" "$seed" "$name" "$((task%2))" \
      > "$OUT/queue/$name.log" 2>&1 &
    pids+=("$!")
    task=$((task+1))
  done
done
for pid in "${pids[@]}"; do wait "$pid"; done
date -Is > "$OUT/queue/encoders.done.txt"
