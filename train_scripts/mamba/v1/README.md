# 静态先验 v1

沿用冻结 B 和 PAVE H1。短训使用 256 张训练图和 512 次更新，评测用上轮未使用的第 51–100 张验证图。[v1 设计](../../../docs/已归档的想法和文件/mamba/RLNN_Static_Prior_v1_Analysis_and_Plan.md) · [实验结果](../../../docs/已归档的想法和文件/mamba/RLNN_Static_Prior_v1_Results.md)

| 组 | 作用 |
| --- | --- |
| `baseline` | 原 B |
| `constants` | 独立训练的加入和移除常量 |
| `centered` | 两个常量加每图均值为零的五维 MLP 节点偏置 |
| `masked` | 复用 `centered` 权重并关闭节点项 |

```bash
export RLNN_SOURCE_SHA="$(git rev-parse HEAD)"
CUDA_VISIBLE_DEVICES=0 bash train_scripts/mamba/v1/train.sh constants &
first=$!
CUDA_VISIBLE_DEVICES=1 bash train_scripts/mamba/v1/train.sh centered &
second=$!
wait "$first"
wait "$second"
bash train_scripts/mamba/v1/eval_matrix.sh
python train_scripts/mamba/v1/summarize.py
```

`eval_matrix.sh` 先并行运行 T100 和 T1000，再并行运行 0.5 秒和 1.0 秒。每个协议只占一张卡。同图内轮转四个臂，限时第二轮反转第一轮的图内顺序。T100 只在末步解码，T1000 和限时协议每十步解码并保留最大的合法候选。单独运行或消融时可用 `bash train_scripts/mamba/v1/eval.sh t1000 --arms baseline centered masked`，并以 `RUN_NAME` 区分输出。完整矩阵写入 `outputs/static_prior_v1`。
