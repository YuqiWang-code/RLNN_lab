#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_NAME="${RUN_NAME:-train_misrb_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${ROOT}/outputs/v000_baseline/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
cd "${ROOT}/models"
exec > >(tee -a "${RUN_DIR}/console.log") 2>&1


python train.py \
  --task "mis" \
  --project_name "consistency_co" \
  --wandb_logger_name "${RUN_NAME}" \
  --do_train \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "${RUN_DIR}" \
  --training_split "data/mis-benchmark-framework/rb/rb200_300_train/*gpickle" \
  --training_split_label_dir "data/mis-benchmark-framework/rb/rb200_300_train_label" \
  --validation_split "data/mis/rb/rb200_300_test/*gpickle" \
  --validation_split_label_dir "data/mis/rb/rb200_300_test_label" \
  --validation_examples 128 \
  --batch_size 4 \
  --num_epochs 50 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 1 \
  --num_workers 64 \
  --boundary_func truncate\
  --alpha 0.3 \
  --consistency \
  --offline \
  "$@"
