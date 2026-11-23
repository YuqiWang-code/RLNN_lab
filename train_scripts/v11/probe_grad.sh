#!/usr/bin/env bash
# v11 C3 gate 1: 梯度自证探针（direction §2.9）
#   20 张验证图、K=20、40 步展开、每 10 步解码，检查五项：
#     1 finite            2 nonzero_grad   3 direction（-g 涨 / +g 掉）
#     4 correlation       5 scale           + feasible（解码仍是独立集）
#   任一项不过 -> 退出码 3，**不许**开 A1（train_scripts/v11/c3.sh）。
# 用法：bash train_scripts/v11/probe_grad.sh [gpu] [batch]
#   gpu   默认 1（GPU0 是别人的，不许碰）
#   batch 单进程摊薄引擎 SHARDS=1，本脚本不并行
set -uo pipefail
cd /home/hzeng/project/ZH/RLNN_lab || exit 1
export OMP_NUM_THREADS=1
GPU=${1:-1}
PY=/home/hzeng/envs/rlnn/bin/python
OUT=outputs/v11
mkdir -p "$OUT" tmp

NAME=probe_unroll_grad
echo "=== [$(date +%H:%M:%S)] $NAME gpu=$GPU ==="
CUDA_VISIBLE_DEVICES=$GPU $PY -u models/probe_unroll_grad.py \
  --split validation --graphs 20 --steps 40 --decode_every 10 \
  --num_k 20 --probe_seed 2027 \
  --out "$OUT/probe_unroll_grad.json" \
  > tmp/$NAME.log 2>&1
rc=$?
echo "--- $NAME exit=$rc ---"
tail -40 tmp/$NAME.log
exit $rc
