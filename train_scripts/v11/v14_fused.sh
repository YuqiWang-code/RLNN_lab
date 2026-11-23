#!/usr/bin/env bash
# t21 / P1：融合后的 0.2 s 臂（与 B 同口径，只多了 torch.compile 融合），三 seed × 1000 图。
#
#   B  （无融合） = outputs/v11/t7/v13_nopave_02s_s${seed}/      （只读复用）
#   B+ （融合）   = outputs/v11/t7/v13_fused_nopave_02s_s${seed}/
#
# 两臂同为 --skip_incumbent --no_pave（整段预算给搜索、pave=false），差异只有 --compile_model；
# run.json / v11_rows.csv 记 compile_model=0|1，便于事后分表，绝不混进同一均值表。
# 已有 CSV 即 SKIP（绝不覆盖）。用法：bash train_scripts/v11/v14_fused.sh [GPU]
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
        --budgets 0.2 --skip_incumbent --no_pave --compile_model)

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

for seed in 1017 1019 1021; do
  if [ -e "$OUT/v13_nopave_02s_s${seed}/padded/padded_ref0.csv" ]; then
    echo "=== 臂 B seed $seed 复用 P0.5 产物 v13_nopave_02s_s${seed} ==="
  else
    echo "!!! 臂 B seed $seed 缺失：先跑 train_scripts/v11/v14_ab.sh" >&2
  fi
done

for seed in 1017 1019 1021; do
  run_cell "v13_fused_nopave_02s_s${seed}" --seeds "$seed"
done

echo "=== [$(date +%H:%M:%S)] P1 融合 0.2 s 三 seed 完成 ==="
for d in $OUT/v13_fused_nopave_02s_s*; do
  echo "  $d $(wc -l < $d/padded/padded_ref0.csv 2>/dev/null) rows"
done
