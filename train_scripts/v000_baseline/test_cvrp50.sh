#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_NAME="${RUN_NAME:-test_cvrp50_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${ROOT}/outputs/v000_baseline/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
cd "${ROOT}/models"
exec > >(tee -a "${RUN_DIR}/console.log") 2>&1


python train.py \
  --task "cvrp" \
  --project_name "consistency_co_test" \
  --wandb_logger_name "${RUN_NAME}" \
  --do_test --storage_path "${RUN_DIR}" \
  --test_split "data/test_dataset/cvrp/cvrp50_hgs-1s_10.366.txt" \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 1 \
  --two_opt_iterations 1 \
  --ckpt_path "ckpts/cvrp50.ckpt" \
  --consistency --resume_weight_only --parallel_sampling 1 --sequential_sampling 1 \
  --rewrite --rewrite_steps 5 --rewrite_ratio 0.5 --m1 0.5 --m2 0.7 --offline \
  "$@"
