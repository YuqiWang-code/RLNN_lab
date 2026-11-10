#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
mkdir -p "$OUT/queue"
case "${1:-collect-train}" in
  collect-train)
    pids=()
    for horizon in 1 2; do
      WORKERS=24 bash train_scripts/legacy/v3/collect.sh --starts \
        "$OUT/starts_train_seed0_stream0/starts.pt" "$OUT/starts_train_seed0_stream1/starts.pt" \
        --horizon "$horizon" --result-dir "$OUT/labels_train_h$horizon" \
        > "$OUT/queue/labels_train_h$horizon.log" 2>&1 &
      pids+=("$!")
      WORKERS=12 bash train_scripts/legacy/v3/collect.sh --starts \
        "$OUT/starts_validation_seed0_stream0/starts.pt" "$OUT/starts_validation_seed1_stream0/starts.pt" \
        "$OUT/starts_validation_seed2_stream0/starts.pt" --horizon "$horizon" \
        --result-dir "$OUT/labels_validation_h$horizon" > "$OUT/queue/labels_validation_h$horizon.log" 2>&1 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    pids=()
    for horizon in 1 2; do
      decision="$OUT/decision_h$horizon.json"
      if [[ "$horizon" == 2 ]]; then decision="$OUT/decision.json"; fi
      "$PYTHON" train_scripts/legacy/v3/decide.py --validation "$OUT/census" \
        --training "$OUT/labels_train_h$horizon" --validation-data "$OUT/labels_validation_h$horizon/states.pt" \
        --variant "D$((horizon+1))_neutral_h$horizon" --output "$decision"
      if "$PYTHON" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["train"] is True else 1)' "$decision"; then
        for seed in 0 1 2; do
          name="stats_h${horizon}_seed${seed}"
          DECISION="$decision" bash train_scripts/legacy/v3/train.sh "$OUT/labels_train_h$horizon/states.pt" \
            "$OUT/labels_validation_h$horizon/states.pt" stats "$seed" "$name" "$(((seed+horizon)%2))" \
            > "$OUT/queue/$name.log" 2>&1 &
          pids+=("$!")
        done
      fi
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    date -Is > "$OUT/queue/stats.done.txt"
    ;;
  *) echo "Usage: $0 collect-train" >&2; exit 2 ;;
esac
