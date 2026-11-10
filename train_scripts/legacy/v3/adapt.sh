#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
bash train_scripts/legacy/v3/collect.sh --starts "$OUT/starts_train_seed0_stream0/starts.pt" \
  "$OUT/starts_train_seed0_stream1/starts.pt" --horizon 1 --policy pave \
  --value-checkpoint "$OUT/stats_h1_seed0/checkpoints/best.pt" --result-dir "$OUT/labels_train_pave_h1"
"$PYTHON" train_scripts/legacy/v3/mix_labels.py --original "$OUT/labels_train_h1" \
  --policy "$OUT/labels_train_pave_h1" --result-dir "$OUT/labels_train_mixed_h1" --seed 0
"$PYTHON" train_scripts/legacy/v3/decide.py --validation "$OUT/census" \
  --training "$OUT/labels_train_mixed_h1" --validation-data "$OUT/labels_validation_h1/states.pt" \
  --variant D2_neutral_h1 --output "$OUT/decision_adapted.json"
if "$PYTHON" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["train"] is True else 1)' "$OUT/decision_adapted.json"; then
  pids=()
  for seed in 0 1 2; do
    DECISION="$OUT/decision_adapted.json" bash train_scripts/legacy/v3/train.sh \
      "$OUT/labels_train_mixed_h1/states.pt" "$OUT/labels_validation_h1/states.pt" \
      stats "$seed" "stats_adapted_h1_seed$seed" "$((seed%2))" > "$OUT/queue/stats_adapted_h1_seed$seed.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
fi
date -Is > "$OUT/queue/adapted.done.txt"
