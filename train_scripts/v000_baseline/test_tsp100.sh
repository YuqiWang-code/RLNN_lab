#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_NAME="${RUN_NAME:-test_tsp100_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${ROOT}/outputs/v000_baseline/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
cd "${ROOT}/models"
exec > >(tee -a "${RUN_DIR}/console.log") 2>&1

TASK="tsp"
PROJECT_NAME="consistency_co_test_grid_tsp100"
TEST_SPLIT="data/tsp_test/tsp100_concorde_7.75585.txt"
CKPT_PATH="ckpts/tsp100.ckpt"
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
  --resume_weight_only \
  --parallel_sampling 1 \
  --sequential_sampling 1 \
  --rewrite \
  --save_tsp_rewrite_trace \
  --tsp_rewrite_trace_dir "${RUN_DIR}/trace" \
  --rewrite_steps 5 \
  --rewrite_iterations 5 \
  --rewrite_ratio 0.1 \
  --m1 0.9 \
  --m2 0.7 \
  --offline \
  "$@"
