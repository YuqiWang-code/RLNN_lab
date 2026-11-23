#!/usr/bin/env bash
# t21 / P0.5 补格：0.1 s 档 PAVE on/off 对撞，三 seed × 1000 图，全部写新目录。
#
#   臂 A'  前导解码 off + PAVE on  = outputs/v11/t7/v13_noprelude_01s_s${seed}/
#   臂 B'  前导解码 off + PAVE off = outputs/v11/t7/v13_nopave_01s_s${seed}/
#
# 现况参照 C'（前导解码 on + PAVE on）只有 seed 1017 的 t7 ladder 产物
# （outputs/v11/t7/ladder_s1017_1000，0.1 s 行），本脚本只读引用、不重跑。
# 目的：检验「PAVE 在可继续搜索的预算区间是否恒被支配」，并为
# G(C) > K×slope 提供 0.1 s 档的实测点（t4 只给过均值 C=0.511）。
# 已有 CSV 即 SKIP（绝不覆盖）。用法：bash train_scripts/v11/v14_ab01.sh [GPU]
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
        --budgets 0.1)

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

if [ -e "$OUT/ladder_s1017_1000/padded/padded_ref0.csv" ]; then
  echo "=== 臂 C' 只读引用 outputs/v11/t7/ladder_s1017_1000（0.1 s 行，seed 1017）==="
else
  echo "!!! 臂 C' 缺失：需要 t7 ladder 产物（仅 seed 1017）" >&2
fi

# 臂 A'：前导解码 off，PAVE on
for seed in 1017 1019 1021; do
  run_cell "v13_noprelude_01s_s${seed}" --seeds "$seed" --skip_incumbent
done

# 臂 B'：前导解码 off + PAVE off（reserve 强制 0，整段预算给搜索；final ≡ search）
for seed in 1017 1019 1021; do
  run_cell "v13_nopave_01s_s${seed}" --seeds "$seed" --skip_incumbent --no_pave
done

echo "=== [$(date +%H:%M:%S)] v14 0.1 s A'/B' 完成 ==="
for d in $OUT/v13_noprelude_01s_s* $OUT/v13_nopave_01s_s*; do
  echo "  $d $(wc -l < $d/padded/padded_ref0.csv 2>/dev/null) rows"
done
