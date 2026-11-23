#!/usr/bin/env bash
# M6 K 扫描（部署口径）：只改 num_k，其余逐项相同。
#   部署口径 = padded/captured + dense + fp16 + 1.0 s deadline + seed 1017
#             + warm start outputs/v2/none_seed0/checkpoints/best.pt
#             + --decode_every 1000 --decode_top 4（末端解码 + top4，即 808 步 / 1.179 ms 那套）
# 图集：官方 manifest 顺序的 idx 0..199（--test_samples 200，规则固定可复现）
# 用法：bash train_scripts/v11/validation/k_sweep_v11.sh <gpu>
set -uo pipefail
cd /home/hzeng/project/ZH/RLNN_lab || exit 1
export OMP_NUM_THREADS=1
GPU=${1:-1}
PY=/home/hzeng/envs/rlnn/bin/python
OUT=outputs/v11_validation

for K in 20 50 100 200; do
  NAME=k${K}_n200_1s
  echo "=== [$(date +%H:%M:%S)] $NAME ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY -u models/eval_v8_padded.py \
    --test_path models/data/MIS_tensors/RB_small_author_test --test_samples 200 \
    --shards 1 --shard 0 --budgets 1.0 --seeds 1017 --reference_seed 0 \
    --save_dir outputs/v2/none_seed0/checkpoints --num_k $K --decoders captured \
    --decode_every 1000 --decode_top 4 --dense \
    --result_dir /home/hzeng/project/ZH/RLNN_lab/$OUT/$NAME \
    > tmp/$NAME.log 2>&1
  echo "--- $NAME exit=$? ---"
  tail -3 tmp/$NAME.log
done

# 步数匹配对照：固定 808 步（与部署 1 s 实际步数一致），K ∈ {20,200}
for K in 20 200; do
  NAME=steps808_k${K}_n200
  echo "=== [$(date +%H:%M:%S)] $NAME ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY -u train_scripts/v11/validation/k_sweep_steps.py \
    --num_k $K --fixed_steps 808 --test_samples 200 --seeds 1017 \
    --save_dir outputs/v2/none_seed0/checkpoints --dense --decode_top 4 \
    --result_dir /home/hzeng/project/ZH/RLNN_lab/$OUT/$NAME \
    > tmp/$NAME.log 2>&1
  echo "--- $NAME exit=$? ---"
  tail -3 tmp/$NAME.log
done
echo "ALL DONE $(date +%H:%M:%S)"
