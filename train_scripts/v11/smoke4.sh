#!/usr/bin/env bash
# v11 4 图冒烟：把「engine 硬闸门 + 新 --fixed_steps 透传 + 逐图 CSV」跑通，
# 不追求数字，只要求全部断言绿。
#   1) models/mis_v11_gates.py engine  --graphs 4
#      eager_step vs captured 逐位 / unroll vs engine 逐位 / 解码可行 /
#      deadline 语义（每块 synchronize 后才判） / max_blocks -> steps 恰为 blocks*chunk /
#      同 seed 可复现
#   2) eval_v8_padded.py --test_samples 4 --budgets 1.0 0.2（部署口径，含 PAVE）
#   3) eval_v8_padded.py --fixed_steps 810（验证 §2.10 透传；steps 必须正好 810）
# 用法：bash train_scripts/v11/smoke4.sh [gpu]
set -uo pipefail
cd /home/hzeng/project/ZH/RLNN_lab || exit 1
export OMP_NUM_THREADS=1
GPU=${1:-1}
PY=/home/hzeng/envs/rlnn/bin/python
OUT=outputs/v11/smoke
mkdir -p "$OUT/timing" "$OUT/fixed" tmp
RC=0

echo "=== [$(date +%H:%M:%S)] 1/3 engine gates (4 graphs) ==="
CUDA_VISIBLE_DEVICES=$GPU $PY -u models/mis_v11_gates.py engine \
  --graphs 4 --dense 1 --chunk 10 --decode_chunk 4 --rounds 40 \
  --test_path models/data/MIS_tensors/RB_small_author_test --split test \
  --save_dir outputs/v2/none_seed0/checkpoints --out "$OUT" \
  > tmp/smoke_engine.log 2>&1 || RC=1
tail -30 tmp/smoke_engine.log

echo "=== [$(date +%H:%M:%S)] 2/3 eval 4 graphs, deployment caliber ==="
CUDA_VISIBLE_DEVICES=$GPU $PY -u models/eval_v8_padded.py \
  --test_path models/data/MIS_tensors/RB_small_author_test --test_samples 4 \
  --shards 1 --shard 0 --budgets 1.0 0.2 --seeds 1017 --reference_seed 0 --repeats 1 \
  --save_dir outputs/v2/none_seed0/checkpoints --num_k 200 \
  --decoders captured cached --decode_every 1000 --decode_top 4 --dense \
  --result_dir /home/hzeng/project/ZH/RLNN_lab/$OUT/timing \
  --v11_csv /home/hzeng/project/ZH/RLNN_lab/$OUT/timing/v11_rows.csv \
  > tmp/smoke_eval.log 2>&1 || RC=1
tail -12 tmp/smoke_eval.log

echo "=== [$(date +%H:%M:%S)] 3/3 eval 4 graphs, --fixed_steps 810 ==="
CUDA_VISIBLE_DEVICES=$GPU $PY -u models/eval_v8_padded.py \
  --test_path models/data/MIS_tensors/RB_small_author_test --test_samples 4 \
  --shards 1 --shard 0 --budgets 1.0 --seeds 1017 --reference_seed 0 --repeats 1 \
  --save_dir outputs/v2/none_seed0/checkpoints --num_k 200 \
  --decoders captured --decode_every 1000 --decode_top 4 --dense --fixed_steps 810 \
  --result_dir /home/hzeng/project/ZH/RLNN_lab/$OUT/fixed \
  --v11_csv /home/hzeng/project/ZH/RLNN_lab/$OUT/fixed/v11_rows.csv \
  > tmp/smoke_fixed.log 2>&1 || RC=1
tail -8 tmp/smoke_fixed.log

# steps 必须正好是 ceil(810/10)*10 = 810
$PY - "$OUT/fixed/v11_rows.csv" <<'PY' || RC=1
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding='utf-8')))
steps = sorted({int(float(r['steps'])) for r in rows})
print(f"fixed_steps check: rows={len(rows)} steps={steps}")
if steps != [810]:
    print("FAIL: --fixed_steps 810 did not pin the step count")
    raise SystemExit(1)
print("OK: fixed_steps pinned")
PY

echo "=== smoke rc=$RC ==="
exit $RC
