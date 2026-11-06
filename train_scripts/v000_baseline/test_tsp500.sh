#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_NAME="${RUN_NAME:-test_tsp500_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${ROOT}/outputs/v000_baseline/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
cd "${ROOT}/models"
exec > >(tee -a "${RUN_DIR}/console.log") 2>&1

TASK="tsp"
PROJECT_NAME="consistency_co_budget300_tsp500"
TEST_SPLIT="data/tsp_test/tsp500_lkh_50000_16.54811.txt"
CKPT_PATH="ckpts/tso500.ckpt"
python train.py \
  --task "${TASK}" \
  --project_name "${PROJECT_NAME}" \
  --wandb_logger_name "${RUN_NAME}" \
  --do_test \
  --storage_path "${RUN_DIR}" \
  --test_split "${TEST_SPLIT}" \
  --two_opt_iterations 5000 \
  --ckpt_path "${CKPT_PATH}" \
  --consistency \
  --sparse_factor 50 \
  --resume_weight_only \
  --parallel_sampling 1 \
  --sequential_sampling 1 \
  --rewrite \
  --rewrite_steps 5 \
  --rewrite_iterations 5 \
  --rewrite_ratio 0.2 \
  --m1 0.9 \
  --m2 0.7 \
  --offline \
  "$@"
