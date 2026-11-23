#!/usr/bin/env bash
# t21 收尾：1 s 档的三条 deadline 臂（captain 2026-11-18 指令）。
#
#   L1 = 1 s deadline + 前导解码关 + PAVE on
#   L2 = 1 s deadline + 前导解码关 + PAVE off
#   L3 = 1 s deadline + 前导解码关 + PAVE off + torch.compile 融合（PAVE 按 L1/L2 胜者）
#
# 全部 deadline 口径（不用 --fixed_steps），因此报的是实际步数；A0 的 810 步不在本脚本里改。
# 每个 cell 都是新目录，已有 CSV 即 SKIP（绝不覆盖）。
# 用法：bash train_scripts/v11/v14_l1l2.sh [GPU] [seeds...]
set -uo pipefail
cd /home/hzeng/project/ZH/RLNN_lab || exit 1
export OMP_NUM_THREADS=1
GPU=${1:-1}; shift || true
SEEDS=("${@:-}")
if [ -z "${SEEDS[0]:-}" ]; then SEEDS=(1017 1019 1021); fi
PY=/home/hzeng/envs/rlnn/bin/python
OUT=outputs/v11/t7
mkdir -p "$OUT" tmp

COMMON=(--test_path models/data/MIS_tensors/RB_small_author_test --test_samples 1000
        --shards 1 --shard 0 --reference_seed 0
        --save_dir outputs/v2/none_seed0/checkpoints --num_k 200 --decoders captured
        --decode_every 1000 --decode_top 4 --dense --mixed_precision fp16 --repeats 1
        --budgets 1.0 --skip_incumbent)

run_cell () {
  local name=$1; shift
  local dir=/home/hzeng/project/ZH/RLNN_lab/$OUT/$name
  if [ -e "$dir/padded/padded_ref0.csv" ]; then
    echo "=== $name SKIP (artifact exists, never overwrite) ==="
    return 0
  fi
  echo "=== [$(date +%H:%M:%S)] $name ==="
  mkdir -p "$dir"
  CUDA_VISIBLE_DEVICES=$GPU $PY -u models/eval_v8_padded.py "${COMMON[@]}" "$@" \
    --result_dir "$dir" --v11_csv "$dir/v11_rows.csv" > tmp/v14_$name.log 2>&1
  local rc=$?
  echo "--- $name exit=$rc ($(date +%H:%M:%S)) ---"
  tail -1 tmp/v14_$name.log
  return $rc
}

for seed in "${SEEDS[@]}"; do
  run_cell "v13_L1_1s_s${seed}"  --seeds "$seed"
  run_cell "v13_L2_1s_s${seed}"  --seeds "$seed" --no_pave --reserve_ms 15
  run_cell "v13_L3_1s_s${seed}"  --seeds "$seed" --no_pave --reserve_ms 15 --compile_model
done

echo "=== [$(date +%H:%M:%S)] 1 s 三臂完成（seeds: ${SEEDS[*]}）==="
for d in $OUT/v13_L?_1s_s*/; do
  echo "  $d $(wc -l < $d/padded/padded_ref0.csv 2>/dev/null || echo 0) rows"
done
