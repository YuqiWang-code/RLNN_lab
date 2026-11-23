#!/usr/bin/env bash
# t21 / P0.5：0.2 s 档三臂对撞（前导解码 × PAVE），三 seed × 1000 图，全部写新目录。
#
#   臂 C（现况参照） 前导解码 on  + PAVE on  = outputs/v11/t7/a02_s${seed}_1000/（t7 产物，只读复用）
#   臂 A             前导解码 off + PAVE on  = outputs/v11/t7/v13_noprelude_02s_s${seed}/
#   臂 B             前导解码 off + PAVE off = outputs/v11/t7/v13_nopave_02s_s${seed}/
#
# A vs B 给出「PAVE 的 ~29 ms 换成搜索时间值多少分」的兑换率（同一预算/seed/图集）。
# 开关都是 additive（models/eval_v8_padded.py --skip_incumbent / --no_pave），默认行为不变；
# run.json 记 leading_decode=on|off 与 pave=true|false，v11_rows.csv 同名列，便于事后分表。
# 已有 CSV 即 SKIP（绝不覆盖）。用法：bash train_scripts/v11/v14_ab.sh [GPU]
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
        --decode_every 1000 --decode_top 4 --dense --mixed_precision fp16 --repeats 1
        --budgets 0.2)

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
  tail -2 tmp/v14_$name.log
  return $rc
}

# 臂 C 只核对存在性（read-only，不重跑）
for seed in 1017 1019 1021; do
  if [ -e "$OUT/a02_s${seed}_1000/padded/padded_ref0.csv" ]; then
    echo "=== 臂 C seed $seed 复用 t7 只读产物 a02_s${seed}_1000 ==="
  else
    echo "!!! 臂 C seed $seed 缺失：先跑 train_scripts/v11/t7_runs.sh" >&2
  fi
done

# 臂 A：前导解码 off，PAVE on
for seed in 1017 1019 1021; do
  run_cell "v13_noprelude_02s_s${seed}" --seeds "$seed" --skip_incumbent
done

# 臂 B：前导解码 off + PAVE off（reserve 强制 0，整段预算给搜索；final ≡ search）
for seed in 1017 1019 1021; do
  run_cell "v13_nopave_02s_s${seed}" --seeds "$seed" --skip_incumbent --no_pave
done

echo "=== [$(date +%H:%M:%S)] v14 A/B 完成 ==="
for d in $OUT/v13_noprelude_02s_s* $OUT/v13_nopave_02s_s*; do
  echo "  $d $(wc -l < $d/padded/padded_ref0.csv 2>/dev/null) rows"
done
