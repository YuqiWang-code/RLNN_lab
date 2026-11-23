#!/usr/bin/env bash
# t7 (A) A0 三 seed 重测 + 质量阶梯（P1 律复核的输入档）
#   声明配置：padded/captured + dense + fp16 + K=200 + decode_every=1000 + top4
#   1.0 s 档：--fixed_steps 808（81 blocks x chunk 10 = 810 步），判据 = search_size 20.0690 +- 0.01
#   0.2 s 档：deadline 驱动，记录实际步数（不做步数匹配，只作部署档）
#   阶梯档（seed 1017）：0.05 / 0.1 / 0.5 s —— 加上 1.0 s（A0）与 0.2 s（A0 的 0.2 s 档）
#                        就是 P1 律的 5 个搜索档；第 6 档 greedy 由 probe_p1_law_1000.py 在 CPU 上补
# 硬规矩（captain m02418）：每个 run 写**新目录**，目录名带 seed/用途，禁止复用已有目录名；
#   outputs/v11/gate0/a0_steps808_n1000/ 自此只读。
# 用法：bash train_scripts/v11/t7_seeds.sh [gpu]
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
        --decode_every 1000 --decode_top 4 --dense --mixed_precision fp16)

run_cell () {                       # run_cell <name> <extra flags...>
  local name=$1; shift
  local dir=/home/hzeng/project/ZH/RLNN_lab/$OUT/$name
  if [ -e "$dir/padded/padded_ref0.csv" ]; then
    echo "=== $name SKIP: $dir/padded/padded_ref0.csv already exists (never overwrite) ==="
    return 0
  fi
  echo "=== [$(date +%H:%M:%S)] $name ==="
  mkdir -p "$dir"
  CUDA_VISIBLE_DEVICES=$GPU $PY -u models/eval_v8_padded.py "${COMMON[@]}" "$@" \
    --result_dir "$dir" --v11_csv "$dir/v11_rows.csv" > tmp/t7_$name.log 2>&1
  local rc=$?
  echo "--- $name exit=$rc ---"
  tail -3 tmp/t7_$name.log
  return $rc
}

# ---------------------------------------------------------------- (A) 1.0 s，三 seed
for seed in 1017 1019 1021; do
  run_cell "a0_s${seed}_steps808_1000" --budgets 1.0 --seeds "$seed" --fixed_steps 808
done

# ---------------------------------------------------------------- (A) 0.2 s，三 seed（deadline）
for seed in 1017 1019 1021; do
  run_cell "a02_s${seed}_1000" --budgets 0.2 --seeds "$seed"
done

# ------------------------------------------------------- (B) 阶梯档（seed 1017，补 0.05/0.1/0.5）
run_cell "ladder_s1017_1000" --budgets 0.05 0.1 0.5 --seeds 1017

echo "=== [$(date +%H:%M:%S)] t7 seeds+ 完成 ==="
ls -d $OUT/*/ | sed 's|^|  |'
