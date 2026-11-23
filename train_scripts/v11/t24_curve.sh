#!/usr/bin/env bash
# t24 定标：K=200 / fp16+dense 的步数-质量曲线（固定步数、PAVE 预留关掉）
#   步数档 130 / 200 / 280 / 350 / 450，每档 1000 图、seed 1017、padded_short 口径。
#   配置与 t7 的 0.2 s 档（P0）同口径，唯一差别：cutoff 换成固定步数、reserve 置 0
#   （`--reserve_frac 0` ⇒ PAVE 只拿到 0 秒，final_size 应等于 search_size）。
#   每档写新目录 outputs/v11/t7/curve_k200_s1017_stepNNN/；目录里已有 CSV 就 SKIP，绝不覆盖。
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
        --budgets 1.0 --seeds 1017 --reserve_frac 0)

for steps in 130 200 280 350 450; do
  name="curve_k200_s1017_step${steps}"
  dir=/home/hzeng/project/ZH/RLNN_lab/$OUT/$name
  if [ -e "$dir/padded/padded_ref0.csv" ]; then
    echo "=== $name SKIP (artifact exists, never overwrite) ==="
    continue
  fi
  echo "=== [$(date +%H:%M:%S)] $name ==="
  mkdir -p "$dir"
  CUDA_VISIBLE_DEVICES=$GPU $PY -u models/eval_v8_padded.py "${COMMON[@]}" \
    --fixed_steps "$steps" --result_dir "$dir" --v11_csv "$dir/v11_rows.csv" \
    > "tmp/t24_${name}.log" 2>&1
  echo "--- $name exit=$? ($(date +%H:%M:%S)) ---"
  tail -2 "tmp/t24_${name}.log"
done

echo "=== [$(date +%H:%M:%S)] t24 曲线完成 ==="
for steps in 130 200 280 350 450; do
  f="$OUT/curve_k200_s1017_step${steps}/padded/padded_ref0.csv"
  echo "  step${steps}: $(wc -l < "$f" 2>/dev/null) rows"
done
