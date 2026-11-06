#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_NAME="${RUN_NAME:-train_tsp100_$(date +%Y%m%d_%H%M%S)}"
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
  --learning_rate 0.001 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "${RUN_DIR}" \
  --training_split "data/tsp_train/tsp100_train_concorde.txt" \
  --validation_split "data/tsp/tsp100_concorde.txt" \
  --test_split "data/tsp_test/tsp100_concorde_7.75585.txt" \
  --batch_size 12 \
  --num_epochs 50 \
  --validation_examples 1280 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 1 \
  --two_opt_iterations 0 \
  --alpha 0.5 \
  --boundary_func truncate \
  --consistency \
  --offline \
  "$@"
