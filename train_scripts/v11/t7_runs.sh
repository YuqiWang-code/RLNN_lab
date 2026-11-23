#!/usr/bin/env bash
# t7 精简执行计划（repeats=1；captain 金额上限 1.3 GPU·h，实测吞吐 ~27 图/分钟）
#   seed 1017 的 1.0 s A0 **不重跑**：直接复用门 0 的只读 artifact
#   outputs/v11/gate0/a0_steps808_n1000/（同配置同 seed，repeats=2，20.0690）。
#   固定步数模式是确定性的（门 0 两次运行 1000/1000 逐位相同），故 repeats=1 与 repeats=2
#   在同一 seed 上等价；补跑的 1019/1021 用 repeats=1 以控成本。
# 全部写新目录；已有目录里的 CSV 存在即 SKIP（绝不覆盖）。
set -uo pipefail
cd /home/hzeng/project/ZH/RLNN_lab || exit 1
export OMP_NUM_THREADS=1
GPU=${1:-1}
PY=/home/hzeng/envs/rlnn/bin/python
OUT=outputs/v11/t7
mkdir -p "$OUT" tmp

COMMON=(--test_path models/data/MIS_tensors/RB_small_author_test --test_samples 1000
        --shards 1 --shard 0 --reference_seed 0
        --save_dir outputs/v2/none_seed0/checkpoints --num_k 200 --decoders captured
        --decode_every 1000 --decode_top 4 --dense --mixed_precision fp16 --repeats 1)

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
    --result_dir "$dir" --v11_csv "$dir/v11_rows.csv" > tmp/t7_$name.log 2>&1
  local rc=$?
  echo "--- $name exit=$rc ($(date +%H:%M:%S)) ---"
  tail -2 tmp/t7_$name.log
  return $rc
}

# (A) 1.0 s 三 seed：1017 复用门 0；这里补 1019 / 1021
for seed in 1019 1021; do
  run_cell "a0_s${seed}_steps808_1000" --budgets 1.0 --seeds "$seed" --fixed_steps 808
done

# (A) 0.2 s 三 seed（deadline 驱动，记录实际步数）
for seed in 1017 1019 1021; do
  run_cell "a02_s${seed}_1000" --budgets 0.2 --seeds "$seed"
done

# (B) 阶梯档：0.1 s 与 0.5 s（0.05 s 是 t4 认定的伪档「只有 5 步、prelude 主导」，不再跑；
#     1.0 s 与 0.2 s 档已由上面的 A 组提供）
run_cell "ladder_s1017_1000" --budgets 0.1 0.5 --seeds 1017

echo "=== [$(date +%H:%M:%S)] t7 runs 完成 ==="
for d in $OUT/*/; do echo "  $d $(wc -l < $d/padded/padded_ref0.csv 2>/dev/null) rows"; done
