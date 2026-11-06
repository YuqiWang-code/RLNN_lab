#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="/home/hzeng/envs/nexco/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
RUN_NAME="${RUN_NAME:-tsp100_train_seed1234_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${ROOT}/outputs/v001_tsp100/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
cd "${ROOT}/models"
exec > >(tee "${RUN_DIR}/console.log") 2>&1
python train.py \
  --task tsp --model_type gcn --seed 1234 \
  --project_name nexco_tsp100 --wandb_logger_name "${RUN_NAME}" \
  --storage_path "${RUN_DIR}" --do_train --offline \
  --training_split data/tsp_train/tsp100_train_lkh100_10000.txt \
  --validation_split data/tsp/tsp100_val_lkh100_128.txt \
  --batch_size 12 --num_epochs 1 --num_workers 4 --validation_examples 128 \
  --n_layers 12 --hidden_dim 256 \
  --learning_rate 0.001 --weight_decay 0.0001 --lr_scheduler cosine-decay \
  --consistency --alpha 0.5 --boundary_func truncate \
  --rewrite_steps 1 --rewrite_iterations 1 --rewrite_ratio 0.25 \
  --m1 0 --m2 0 --two_opt_iterations 0 \
  "$@"
