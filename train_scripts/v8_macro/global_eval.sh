#!/usr/bin/env bash
# Evaluate the global-token backbone against the frozen one, same harness.
#
# The frozen 1.0 s reference on the 1000 author graphs is 19.9333
# (captured/cached); the roadmap threshold for keeping the global token is
# 20.00 at 1.0 s.
set -euo pipefail
cd "${RLNN_ROOT:-/home/hzeng/project/ZH/RLNN_lab}" || exit 1
source train_scripts/v8_macro/common.sh
export CUDA_VISIBLE_DEVICES="${V8_GPU:-1}"

RUN_NAME="${RUN_NAME:-global_tp50}"
GRAPHS="${GRAPHS:-500}"
BUDGETS="${BUDGETS:-0.5 1.0}"
RESULT_DIR="${RESULT_DIR:-$OUT/global_eval}"
ARCH="${ARCH:-global}"
SEEDS_LIST="${SEEDS_LIST:-1017}"

CHECKPOINT="$ROOT/outputs/v2/$RUN_NAME/checkpoints"
if [ ! -f "$CHECKPOINT/best.pt" ]; then
    echo "missing checkpoint: $CHECKPOINT/best.pt" >&2
    exit 1
fi

mkdir -p "$OUT/queue"
"$PYTHON" "$ROOT/models/eval_v8_capture_budget.py" \
    --split test \
    --test_path "$ROOT/models/data/MIS_tensors/RB_small_author_test" \
    --test_samples "$GRAPHS" \
    --budgets $BUDGETS \
    --arms captured \
    --decoders cached \
    --arch "${ARCH:-global}" \
    --reference_seed "$REFERENCE_SEED" \
    --save_dir "$CHECKPOINT" \
    --seeds $SEEDS_LIST \
    --repeats "${REPEATS:-1}" \
    --decode_chunk "${DECODE_CHUNK:-4}" \
    --result_dir "$RESULT_DIR" 2>&1 | tee "$OUT/queue/global_eval_$RUN_NAME.log"
