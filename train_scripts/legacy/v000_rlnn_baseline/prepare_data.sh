#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/models/data/MIS}"
export PYTHONHASHSEED=0
cd "$ROOT/models/data"
for split in train val test; do
  case "$split" in
    train) count="${TRAIN_COUNT:-1000}"; seed=0 ;;
    val) count="${VALID_COUNT:-100}"; seed=100000 ;;
    test) count="${TEST_COUNT:-500}"; seed=200000 ;;
  esac
  destination="$DATA_ROOT/RB_small_$split"
  if [[ -e "$destination" ]]; then
    echo "Refusing to overwrite existing split: $destination" >&2
    exit 1
  fi
  "${PYTHON:-python}" rb_generator.py --num-graphs "$count" --min-n 200 --max-n 300 \
    --seed "$seed" --save-dir "$destination" --processes "${PROCESSES:-1}"
done
