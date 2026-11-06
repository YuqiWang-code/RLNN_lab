#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_NAME="${RUN_NAME:-test_misrb_large_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${ROOT}/outputs/v000_baseline/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
cd "${ROOT}/models"
exec > >(tee -a "${RUN_DIR}/console.log") 2>&1


python train.py \
  --task "mis" \
  --project_name "consistency_co_test" \
  --wandb_logger_name "${RUN_NAME}" \
  --do_test \
  --storage_path "${RUN_DIR}" \
  --test_split "data/mis-benchmark-framework/rb/rb800_1200_test/*gpickle" \
  --test_split_label_dir "data/mis-benchmark-framework/rb/rb800_1200_test_label" \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 1 \
  --ckpt_path "ckpts/mis_rb_large.ckpt" \
  --consistency \
  --resume_weight_only \
  --parallel_sampling 1 \
  --sequential_sampling 1 \
  --rewrite \
  --rewrite_steps 1 \
  --rewrite_ratio 0.3 \
  --rewrite_iterations 100 \
  --m1 0.2 \
  --offline \
  "$@"
