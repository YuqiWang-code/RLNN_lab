#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="/home/hzeng/envs/nexco/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
RUN_NAME="${RUN_NAME:-tsp100_test_seed1234_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${ROOT}/outputs/v001_tsp100/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
cd "${ROOT}/models"
exec > >(tee "${RUN_DIR}/console.log") 2>&1
python train.py \
  --task tsp --model_type gcn --seed 1234 \
  --project_name nexco_tsp100 --wandb_logger_name "${RUN_NAME}" \
  --storage_path "${RUN_DIR}" --do_test --offline \
  --test_split data/tsp_test/tsp100_concorde_7.75585.txt \
  --ckpt_path "${CKPT_PATH:?Set CKPT_PATH to the trained checkpoint}" \
  --consistency --parallel_sampling 1 --sequential_sampling 1 \
  --rewrite_steps 5 --rewrite_iterations 5 --rewrite_ratio 0.1 \
  --m1 0.9 --m2 0.7 --two_opt_iterations 5000 \
  "$@"
