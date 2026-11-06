#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_NAME="${RUN_NAME:-train_tsp500_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${ROOT}/outputs/v000_baseline/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
cd "${ROOT}/models"
exec > >(tee -a "${RUN_DIR}/console.log") 2>&1


python train.py \
  --task "tsp" \
  --project_name "consistency_co" \
  --wandb_logger_name "${RUN_NAME}" \
  --do_train \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "${RUN_DIR}" \
  --training_split "data/tsp_train/tsp500_uniform_train_lkh5w.txt" \
  --validation_split "data/tsp/tsp500_concorde.txt" \
  --test_split "data/tsp_test/tsp500_lkh_50000_16.54811.txt" \
  --sparse_factor 50 \
  --batch_size 6 \
  --num_epochs 50 \
  --hidden_dim 256 \
  --validation_examples 128 \
  --inference_schedule "cosine" \
  --two_opt_iterations 0 \
  --boundary_func truncate\
  --rewrite_steps 1 \
  --offline \
  --consistency \
  "$@"
