#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_NAME="${RUN_NAME:-train_cvrp50_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${ROOT}/outputs/v000_baseline/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
cd "${ROOT}/models"
exec > >(tee -a "${RUN_DIR}/console.log") 2>&1


python train.py \
  --task "cvrp" \
  --project_name "consistency_co" \
  --wandb_logger_name "${RUN_NAME}" \
  --do_train \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "${RUN_DIR}" \
  --training_split "data/cvrp/cvrp_50/cvrp50_uniform_train.txt" \
  --validation_split "data/test_dataset/cvrp/cvrp50_hgs-1s_10.366.txt" \
  --test_split "data/test_dataset/cvrp/cvrp50_hgs-1s_10.366.txt" \
  --batch_size 32 \
  --num_epochs 50 \
  --hidden_dim 256 \
  --validation_examples 10000 \
  --diffusion_schedule "linear" \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 1 \
  --two_opt_iterations 0 \
  --alpha 0.5 \
  --boundary_func truncate \
  --consistency \
  --offline \
  "$@"
