#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
version=v000_rlnn_baseline
train_run="${RUN_NAME:-rb_train_author_b64_g8_gpu1_seed0_$(date +%Y%m%d_%H%M%S)}"
train_dir="$ROOT/outputs/$version/$train_run"
test="$ROOT/models/data/author_rld4co/MIS/RB_small_test"
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=backend:cudaMallocAsync
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

# An eight-graph batch replaces the eight DDP ranks and averages their losses.
RUN_NAME="$train_run" GRAPH_BATCH_SIZE=8 BATCH_SIZE=64 NUM_WORKERS=8 TEST_PATH="$test" \
  TRAIN_PATH="$ROOT/models/data/MIS_tensors/RB_small_train" \
  VALID_PATH="$ROOT/models/data/MIS_tensors/RB_small_val" \
  bash "$SCRIPT_DIR/train.sh" --seed 0 --epochs 50 --train_samples 1000 --valid_samples 100 \
    --num_t 100 --num_k 20 --num_tp 50 --num_kp 10 --num_d 5 \
    --beta 1.02 --lambd 0.5 --lr 0.0001 --mixed_precision no

# Validation selects the same first strict improvement that saved best.pt.
best_epoch="$("${PYTHON:-python}" - "$train_dir/metrics/metrics.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as stream:
    rows = [row for row in csv.DictReader(stream) if row["val/raw_obj"]]
if len(rows) != 50 or int(rows[-1]["step"]) != 100000:
    raise RuntimeError("expected 50 epochs and 100,000 optimizer updates")
best = min(rows, key=lambda row: float(row["val/raw_obj"]))
print(best["epoch"])
PY
)"
eval_dir="$ROOT/outputs/$version/${train_run}_a3"
mkdir "$eval_dir"
cd "$ROOT/models"
export PYTORCH_ALLOC_CONF=backend:native PYTORCH_CUDA_ALLOC_CONF=backend:native
exec "${PYTHON:-python}" -u eval_author_rb.py --save_dir "$train_dir/checkpoints" \
  --test_path "$test" --result_dir "$eval_dir" --decoder gradient_flip \
  --checkpoint_epoch "$best_epoch" > "$eval_dir/eval.log" 2>&1
