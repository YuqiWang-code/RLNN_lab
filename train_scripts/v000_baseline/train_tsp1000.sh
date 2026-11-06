#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_NAME="${RUN_NAME:-train_tsp1000_$(date +%Y%m%d_%H%M%S)}"
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
  --training_split "data/tsp_train/tsp1000_train_concorde.txt" \
  --validation_split "data/tsp/tsp1000_concorde.txt" \
  --test_split "data/tsp_test/tsp1000_concorde_23.11812.txt" \
  --use_activation_checkpoint \
  --sparse_factor 100 \
  --batch_size 4 \
  --num_epochs 50 \
  --hidden_dim 256 \
  --validation_examples 128 \
  --inference_schedule "cosine" \
  --two_opt_iterations 0 \
  --boundary_func truncate\
  --rewrite_steps 1 \
  --alpha 0.5 \
  --offline \
  --consistency \
  "$@"
