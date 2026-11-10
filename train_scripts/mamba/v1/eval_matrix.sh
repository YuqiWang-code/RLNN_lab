#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

for arm in constants centered; do
  checkpoint="$OUT/train_${arm}_ref${REF_SEED}/training.csv"
  until [[ -f "$checkpoint" ]]; do
    pid="$(cat "$OUT/train_$arm.pid")"
    ps -p "$pid" -o stat= | grep -q '^[^Z ]' || { tail -n 20 "$OUT/train_$arm.log"; exit 1; }
    sleep 3
  done
done
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u "$SCRIPTS/check.py" --root "$OUT" --ref "$REF_SEED"

pair() {
  CUDA_VISIBLE_DEVICES=0 bash "$SCRIPTS/eval.sh" "$1" >"$OUT/$1.log" 2>&1 & first=$!
  CUDA_VISIBLE_DEVICES=1 bash "$SCRIPTS/eval.sh" "$2" >"$OUT/$2.log" 2>&1 & second=$!
  wait "$first"
  wait "$second"
}

pair t100 t1000
date -Is >"$OUT/fixed_completed.txt"
pair budget05 budget10
date -Is >"$OUT/completed.txt"
