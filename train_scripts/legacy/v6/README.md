# v6 SC-RLNN

[模型和验证协议](../../../docs/已归档的想法和文件/RLNN_v6_SC_RLNN.md)

运行前准备与 v5 相同的 RB 张量缓存、三份 B 权重和三份冻结 PAVE Stats H1 权重。环境变量 `REF_SEED=0/1/2` 对应 v6 训练种子 42/123/999。每个任务只使用 `CUDA_VISIBLE_DEVICES` 指定的一张卡。

```bash
export PYTHON=/home/hzeng/envs/rlnn/bin/python
CUDA_VISIBLE_DEVICES=0 REF_SEED=0 EPOCHS=1 RUN_NAME=pilot_full_ref0 \
  bash train_scripts/legacy/v6/train.sh v6_full

CUDA_VISIBLE_DEVICES=0 REF_SEED=0 CHECKPOINT_DIR=outputs/v6/pilot_full_ref0/checkpoints \
  RUN_NAME=pilot_fixed_full_ref0 bash train_scripts/legacy/v6/eval.sh v6_full --cuda-graph

CUDA_VISIBLE_DEVICES=0 "$PYTHON" models/archived_frequency/profile_v6_frequency.py \
  --precision fp16 --cuda-graph --checkpoint outputs/v6/pilot_full_ref0/checkpoints \
  --output outputs/v6/profile_pilot_fp16
```

`eval.sh baseline` 使用 B，`eval.sh v5_control` 使用 DAFR-Lite。`--split test` 指定作者测试集。`--post pave --time-budget-s 0.5` 或 `1.0` 启用对应的连续搜索预算。报告固定预算时，同卡顺序运行不同方法并使用相同执行模式。

`train.sh` 接受 `v6_lowonly`、`v6_noscatter`、`v6_noanti`、`v6_full`，也可单独检查 `v6_highonly` 和 `v6_nogate`。用 `diagnose.sh` 记录分支能量。完整首轮结果同步回来后，`python train_scripts/legacy/v6/summarize.py` 生成逐图配对汇总和准入判断。
