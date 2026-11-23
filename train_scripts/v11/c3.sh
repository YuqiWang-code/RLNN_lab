#!/usr/bin/env bash
# v11 主线 C3 双臂启动器：A1（--loss unroll）与 A1′（--loss erdoes 对照）
#
# 两份 run.json 除 loss 外必须逐字段相同（direction §2.5），审计用
#   python models/mis_v11_gates.py audit --a <A1>/run.json --b <A1p>/run.json
#
# 训练协议（§2.5 写死）：
#   初始化 outputs/v2/none_seed0/checkpoints/best.pt（--warm_start）
#   6 epoch / lr 2e-5 / 训练 K=20 / 展开 200 步每 10 步解码取最后一次反传 / 训练 seed 0
#   **注意**：--graph_batch_size 必须是 1（models/solvers.py:333 规定 loss!=erdoes 只能单图训练），
#   而 --validation_seed / --finetune_lr / --freeze_backbone_epochs 在 gbs<2 时会被
#   models/solvers.py:337 拒掉，所以学习率走 --lr、验证走 solver.evaluate('valid')。
#
# 评测（deployment caliber，与 A0 同一份 eval 代码）：
#   1.0 / 0.2 s × 1000 图 × seed 1017/1019/1021，K=200，captured，dense，fp16
#
# 用法：
#   bash train_scripts/v11/c3.sh [gpu] [unroll|erdoes] [train|eval|all]
#   例：GPU1 训练 + 评测整臂 -> bash train_scripts/v11/c3.sh 1 unroll all
#       只补评测               -> bash train_scripts/v11/c3.sh 1 unroll eval
set -uo pipefail
cd /home/hzeng/project/ZH/RLNN_lab || exit 1
export OMP_NUM_THREADS=1
GPU=${1:-1}
LOSS=${2:-unroll}
PHASE=${3:-all}
# updates per graph inside one epoch.  The frozen erdoes recipe uses 100 (--num_tp),
# which at 200-step unrolls costs ~88 s/graph -> ~24 h for 1000 graphs, i.e. it does
# not fit the 8.2 GPU*h budget of direction §2.6.  TP=2 is the practical setting
# (~3 GPU*h) and must be reported as a protocol deviation when used.
TP=${TP:-2}
PY=/home/hzeng/envs/rlnn/bin/python

case "$LOSS" in
  unroll) ARM=a1_c3;;
  erdoes) ARM=a1p_ctrl;;
  *) echo "loss must be unroll|erdoes"; exit 2;;
esac
TRAIN_DIR=outputs/v11/$ARM
EVAL_DIR=outputs/v11/$ARM/eval
mkdir -p "$TRAIN_DIR" "$EVAL_DIR" tmp

# ------------------------------------------------------------------ 训练
if [ "$PHASE" = train ] || [ "$PHASE" = all ]; then
  NAME=$ARM.train
  echo "=== [$(date +%H:%M:%S)] train $ARM (loss=$LOSS) gpu=$GPU num_tp=$TP ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY -u models/main.py \
    --do_train --method RLNN --problem mis --seed 0 \
    --epochs 6 --lr 2e-5 --batch_size 20 --graph_batch_size 1 \
    --train_path models/data/MIS_tensors/RB_small_train --train_samples 1000 \
    --valid_path models/data/MIS_tensors/RB_small_val --valid_samples 100 \
    --test_path models/data/MIS_tensors/RB_small_author_test \
    --num_workers 4 --num_k 20 --num_t 100 --num_tp "$TP" --num_d 5 --num_l 5 --num_h 128 \
    --mixed_precision no --beta 1.02 --lambd 0.5 \
    --warm_start outputs/v2/none_seed0/checkpoints/best.pt \
    --loss "$LOSS" --unroll_steps 200 --unroll_decode_every 10 --run_json \
    --save_dir "$TRAIN_DIR" \
    > tmp/$NAME.log 2>&1
  rc=$?
  echo "--- train $ARM exit=$rc ---"
  tail -8 tmp/$NAME.log
  grep -o 'unroll steps=[^|]*' tmp/$NAME.log | tail -3
  if [ "$rc" -ne 0 ]; then exit "$rc"; fi
fi

# ------------------------------------------------------------------ 评测
if [ "$PHASE" = eval ] || [ "$PHASE" = all ]; then
  NAME=$ARM.eval
  echo "=== [$(date +%H:%M:%S)] eval $ARM gpu=$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY -u models/eval_v8_padded.py \
    --test_path models/data/MIS_tensors/RB_small_author_test --test_samples 1000 \
    --shards 1 --shard 0 --budgets 1.0 0.2 --seeds 1017 1019 1021 --reference_seed 0 \
    --repeats 1 --save_dir "$TRAIN_DIR" --num_k 200 --decoders captured \
    --decode_every 1000 --decode_top 4 --dense \
    --result_dir /home/hzeng/project/ZH/RLNN_lab/$EVAL_DIR \
    --v11_csv /home/hzeng/project/ZH/RLNN_lab/$EVAL_DIR/v11_rows.csv \
    > tmp/$NAME.log 2>&1
  rc=$?
  echo "--- eval $ARM exit=$rc ---"
  grep -E '^(1\.0|0\.2)s |^wrote ' tmp/$NAME.log | tail -6
  exit "$rc"
fi
