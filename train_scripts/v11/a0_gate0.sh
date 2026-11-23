#!/usr/bin/env bash
# v11 门 0 / A0 双口径（direction §2.7 + §14.4；判据口径按 t11 定稿：R2/R3/R7）
#   (i) 步数匹配口径（确定性主闸门）：padded/captured + dense + fp16 + K=200
#       + 固定 808 步（--fixed_steps 808 -> 81 blocks x chunk 10 = 810 步）
#       + seed 1017 + 1000 图
#       -> mean search_size 20.0690 +- 0.01（声明配置 fp16+dense 的 A0；20.0760 是 fp32 误配，
#          只作 misconfig_fp32_sensitivity = 0.007 的证据，见 docs/temporary/RLNN_v11_Implementation.md §7.6），
#          步数必须恰为 810。
#          dtype / dense / fp32_csr / fixed_steps 一律从 --v11_csv 的**行**读，
#          全部行 100% 一致才算过（R2：不再用 --expect_* 与自己的期望值比对）。
#          final_size 只报告：它含 PAVE 0.2 s 预留的 +-0.0005 抖动（R3）。
#   (ii) 时钟口径（只报告，不当判据）：同一套 flags 走 1.0 s deadline，
#       记录实际步数 / ms per step / load 秒；步数差 >10% 判「机器速度不同」。
# 配置错判据：≈3.9 ms/步、≈244 步 = 跑成 fp32+CSR（--dense 没生效 / 混精度掉了），
#   判配置错，不许开 A1。
# 历史参照：20.0685 是旧 run 的时钟口径数字，其命令未记录、不可审计
#   （historical_clock_reference_unreproducible），只作参考，不作判据（R5/R7）。
# 产物：outputs/v11/gate0/{a0_steps808_n1000,a0_clock1s_n1000}/ 与 gates_gate0*.json
# 用法：bash train_scripts/v11/a0_gate0.sh [gpu]
set -uo pipefail
cd /home/hzeng/project/ZH/RLNN_lab || exit 1
export OMP_NUM_THREADS=1
GPU=${1:-1}
PY=/home/hzeng/envs/rlnn/bin/python
OUT=outputs/v11/gate0
mkdir -p "$OUT" tmp

# 部署口径：配置取自 Paper_Table（K=200、末端解码 + top4、dense、fp16）。
# 注意——旧 run 的命令没有记录，这套 flags 是**从结果表反推**的，不能审计（R5）。
COMMON=(--test_path models/data/MIS_tensors/RB_small_author_test --test_samples 1000
        --shards 1 --shard 0 --budgets 1.0 --seeds 1017 --reference_seed 0
        --save_dir outputs/v2/none_seed0/checkpoints --num_k 200 --decoders captured
        --decode_every 1000 --decode_top 4 --dense --mixed_precision fp16)

# ------------------------------------------------------- 0. 闸门自身必须能失败
echo "=== [$(date +%H:%M:%S)] gates selftest ==="
$PY -u models/mis_v11_gates.py selftest --out "$OUT" || { echo "selftest FAILED"; exit 1; }
echo "=== [$(date +%H:%M:%S)] gate0 negative selftest ==="
$PY -u models/mis_v11_gates.py gate0 --negative-selftest --out "$OUT" \
  || { echo "gate0 negative selftest FAILED (a gate that cannot fail is not a gate)"; exit 1; }

# ---------------------------------------------------------------- (i) 主闸门
NAME=a0_steps808_n1000
echo "=== [$(date +%H:%M:%S)] (i) $NAME (fixed 808 steps) ==="
CUDA_VISIBLE_DEVICES=$GPU $PY -u models/eval_v8_padded.py "${COMMON[@]}" \
  --fixed_steps 808 \
  --result_dir /home/hzeng/project/ZH/RLNN_lab/$OUT/$NAME \
  --v11_csv /home/hzeng/project/ZH/RLNN_lab/$OUT/$NAME/v11_rows.csv \
  > tmp/$NAME.log 2>&1
rc_i=$?
echo "--- (i) $NAME exit=$rc_i ---"
tail -6 tmp/$NAME.log

GATE_I_RC=1
if [ "$rc_i" -eq 0 ]; then
  # gate0 用 legacy CSV 算 search_size / steps，用 v11 CSV 读 dtype/dense/fp32_csr
  CUDA_VISIBLE_DEVICES=$GPU $PY -u models/mis_v11_gates.py gate0 \
    --csv $OUT/$NAME/padded/padded_ref0.csv \
    --json $OUT/$NAME/padded/padded_ref0.json \
    --v11_csv $OUT/$NAME/v11_rows.csv \
    --expect_graphs 1000 --out "$OUT"
  GATE_I_RC=$?
  echo "--- (i) gate0 exit=$GATE_I_RC ---"
fi

# ------------------------------------------------------------ (ii) 只报告
NAME=a0_clock1s_n1000
echo "=== [$(date +%H:%M:%S)] (ii) $NAME (1.0 s deadline, report only) ==="
CUDA_VISIBLE_DEVICES=$GPU $PY -u models/eval_v8_padded.py "${COMMON[@]}" \
  --result_dir /home/hzeng/project/ZH/RLNN_lab/$OUT/$NAME \
  --v11_csv /home/hzeng/project/ZH/RLNN_lab/$OUT/$NAME/v11_rows.csv \
  > tmp/$NAME.log 2>&1
rc_ii=$?
echo "--- (ii) $NAME exit=$rc_ii ---"
tail -6 tmp/$NAME.log
if [ "$rc_ii" -eq 0 ]; then
  # 只打印，不当判据（时钟口径的结果标 steps=，不得混入 A0 对照池）。
  # 它的 steps 天然不等于 810，所以 gate0 在这里必然报 steps_810=False，属预期。
  $PY -u models/mis_v11_gates.py gate0 \
    --csv $OUT/$NAME/padded/padded_ref0.csv \
    --json $OUT/$NAME/padded/padded_ref0.json \
    --v11_csv $OUT/$NAME/v11_rows.csv \
    --expect_graphs 1000 --out "$OUT/clock_report" || true
  echo "--- (ii) 报告完毕（失败不代表门 0 失败，只需比步数与 ms/步） ---"
fi

echo "=== 门 0 判定：主闸门 (i) exit=$GATE_I_RC（0=放行 A1，非 0=停）==="
exit $GATE_I_RC
