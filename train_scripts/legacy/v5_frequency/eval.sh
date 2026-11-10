#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
arm="${1:?pass baseline, analytic, none or a v5 arm}"
shift
seed="${SEED:-0}"
precision="${PRECISION:-no}"
frequency="$arm"
checkpoint="$OUT/train_${arm}_seed${seed}/checkpoints"
analytic=()
if [[ "$arm" == baseline || "$arm" == analytic ]]; then
  frequency=none
  checkpoint="$(base_checkpoint "$seed")"
fi
if [[ "$arm" == analytic ]]; then
  scale="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["analytic_scale"])' "$OUT/calibration_seed$seed/run.json")"
  analytic=(--analytic-scale "$scale")
fi
export PYTHONPATH="$ROOT/models${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -u models/archived_frequency/eval_v5_frequency.py \
  --frequency "$frequency" --save_dir "$checkpoint" --label "${LABEL:-$arm}" \
  --result_dir "$OUT/${RUN_NAME:-eval_fixed_${arm}_seed${seed}_${precision}}" \
  --train_seed "$seed" --seed 1017 --split "${SPLIT:-validation}" --test_samples "${SAMPLES:-100}" \
  --mixed_precision "$precision" --post none pave --horizon 1 --num_k 20 --num_t 100 \
  --num_d 5 --num_l 5 --num_h 128 --method RLNN --problem mis --beta 1.02 --num_workers 0 "${analytic[@]}" "$@"
