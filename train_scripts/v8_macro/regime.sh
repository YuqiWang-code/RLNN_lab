#!/usr/bin/env bash
# Cross-family regime probe: generate ER / BA / large-RB graphs, convert them to the
# tensor cache, then run the three-arm comparison at a fixed clock.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

TENSORS="$ROOT/models/data/MIS_tensors"
RAW="$ROOT/models/data/raw_regime"
GRAPHS="${GRAPHS:-40}"
BUDGETS="${BUDGETS:-1.0}"
FAMILIES="${FAMILIES:-RB_small_val ER_200_300 BA_200_300 RB_large}"
PROCESSES="${PROCESSES:-8}"
PAVE_LIMIT="${PAVE_LIMIT:-}"
KAMIS="${KAMIS:-$ROOT/tmp/kamis_build/redumis}"
RESULT_DIR="${RESULT_DIR:-$OUT/regime}"
mkdir -p "$RESULT_DIR/queue"

cd "$ROOT"
if [ ! -d "$TENSORS/ER_200_300" ]; then
  echo "[regime] generating ER 200-300 (p=0.1)" | tee -a "$RESULT_DIR/queue/regime_gen.log"
  "$PYTHON" models/data/er_generator.py --min-n 200 --max-n 300 --p 0.1 \
    --num-graphs "$GRAPHS" --output-dir "$RAW/ER_200_300" --processes "$PROCESSES" \
    >> "$RESULT_DIR/queue/regime_gen.log" 2>&1
fi
if [ ! -d "$TENSORS/BA_200_300" ]; then
  echo "[regime] generating BA 200-300 (m=4)" | tee -a "$RESULT_DIR/queue/regime_gen.log"
  "$PYTHON" models/data/ba_generator.py --min-n 200 --max-n 300 --m 4 \
    --num-graphs "$GRAPHS" --save-dir "$RAW/BA_200_300" --seed 0 --processes "$PROCESSES" \
    >> "$RESULT_DIR/queue/regime_gen.log" 2>&1
fi
if [ ! -d "$TENSORS/RB_large" ]; then
  echo "[regime] generating RB large (n 800-1400)" | tee -a "$RESULT_DIR/queue/regime_gen.log"
  "$PYTHON" models/data/rb_generator.py --num-graphs "$GRAPHS" --min-n 800 --max-n 1400 \
    --size large --seed 0 --save-dir "$RAW/RB_large" --processes "$PROCESSES" \
    >> "$RESULT_DIR/queue/regime_gen.log" 2>&1
fi

for family in ER_200_300 BA_200_300 RB_large; do
  if [ ! -f "$TENSORS/$family/tensor_manifest.json" ]; then
    echo "[regime] converting $family to tensors" | tee -a "$RESULT_DIR/queue/regime_gen.log"
    "$PYTHON" models/data/prepare_mis_tensors.py --source "$RAW/$family" \
      --destination "$TENSORS/$family" --workers "$PROCESSES" \
      >> "$RESULT_DIR/queue/regime_gen.log" 2>&1
  fi
done

# shellcheck disable=SC2086
PAVE_ARGS=""
if [ -n "$PAVE_LIMIT" ]; then
  PAVE_ARGS="--pave_limit $PAVE_LIMIT"
fi
# shellcheck disable=SC2086
"$PYTHON" -u models/eval_regime.py --families $FAMILIES --graphs "$GRAPHS" \
  --budgets $BUDGETS --seeds ${SEEDS_LIST:-1017} --reference_seed "$REFERENCE_SEED" \
  --kamis "$KAMIS" --result_dir "$RESULT_DIR" $PAVE_ARGS 2>&1 | tee "$RESULT_DIR/queue/regime.log"
