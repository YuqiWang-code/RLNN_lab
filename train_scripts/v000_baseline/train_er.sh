#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_NAME="${RUN_NAME:-train_er_$(date +%Y%m%d_%H%M%S)}"
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
  --training_split "data/mis/er_700_800_train/*gpickle" \
  --training_split_label_dir "data/mis/er_700_800_train_labels" \
  --validation_split "data/mis/er/er_700_800_test/*gpickle" \
  --batch_size 4 \
  --num_epochs 50 \
  --hidden_dim 128 \
  --validation_examples 128 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 1 \
  --use_activation_checkpoint \
  --boundary_func truncate\
  --alpha 0.5 \
  --consistency \
  --offline \
  "$@"
