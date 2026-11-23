# DIMES-MIS 限时复测

四档图集共用在 128 张独立 ER-[700–800] 训练图上训练的检查点。训练直接调用官方的 `GCN_DEEP_DIVER_DIMES` 和 JAX REINFORCE，使用自环、20 层图卷积、64 维隐藏层、单输出热图和一步内层更新。每图用 32 条采样轨迹估计梯度，批大小为 8，训练 4 轮共 64 次外层更新。种子为 32871，预先固定最后一轮权重。原论文的 ER-[700–800] 配方使用 150,000 次外层更新，并在大 ER 上继续训练 200 次。本复测使用较短训练预算，RB 与大 ER 均直接迁移。[官方实现](https://github.com/DIMESTeam/DIMES/tree/95aaa91c74cd47dc441c84eea9d38ead44f61ffd/MIS) · [作者论文附录 D.2](https://q-rz.github.io/static/neurips22/neurips22-dimes-paper.pdf)

训练输入是项目已有的 `models/data/scale_priority_20261121/ER_small/train`。在项目根目录执行下列命令。TensorFlow 使用兼容 v1 的 CPU 执行器，JAX 采样和 PyTorch 测试使用 GPU 1。导出包含一张独立 47 节点图的 TensorFlow 分数和官方 JAX 贪心解，用于检查测试实现。

```bash
export CUDA_VISIBLE_DEVICES=1 CPU_CORE=24
export JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
taskset -c "$CPU_CORE" "$HOME/envs/nexco/bin/python" -u train_scripts/v17/train_dimes.py
taskset -c "$CPU_CORE" "$HOME/envs/nexco/bin/python" -u train_scripts/v17/export_dimes_checkpoint.py
checkpoint=outputs/v17/dimes_training/dimes_er_small.npz
for split in ER_large ER_small RB_large RB_small; do
  bash train_scripts/v17/dimes_200ms.sh "$split" "$checkpoint" "outputs/v17/four_scale_comparators/$split/dimes"
done
```

每个实例从主机边列表开始计时，包括构建稀疏矩阵、GPU 前向和节点集交付。取得分后先按分数贪心解码，再按批生成 32 个受可行性掩码约束的类别样本，保留新出现的更大独立集。测试使用学习到的固定热图；采样用 Gumbel 排序实现该分布。每图按 1017、1019、1021 三个种子独立运行，200 ms 内没有交付的图–种子对记 0。每个发布的节点集在计时后按原图边校验。

| 作者图 | 平均 SIZE | 平均返回（ms） | 按时有解 | 按时返回 |
|---|---:|---:|---:|---:|
| [RB-[200–300]](../../outputs/v17/four_scale_comparators/RB_small/dimes/per_graph.csv) | 18.9183 | 197.54 | 3,000/3,000 | 2,851/3,000 |
| [RB-[800–1200]](../../outputs/v17/four_scale_comparators/RB_large/dimes/per_graph.csv) | 35.4367 | 191.76 | 3,000/3,000 | 2,641/3,000 |
| [ER-[700–800]](../../outputs/v17/four_scale_comparators/ER_small/dimes/per_graph.csv) | 37.4297 | 193.35 | 384/384 | 351/384 |
| [ER-[9000–11000]](../../outputs/v17/four_scale_comparators/ER_large/dimes/per_graph.csv) | 178.8542 | 279.00 | 30/48 | 3/48 |

在 16 张独立小 ER 验证图上，最终检查点的按时 SIZE 为 37.0625，16/16 次按时有解。导出的 47 节点对拍图上，PyTorch 与官方 TensorFlow 的分数最大绝对误差为 0.00002313，两者经贪心解码得到同一组 12 个节点。[训练配置](../../outputs/v17/dimes_training/run.json) · [冻结权重](../../outputs/v17/dimes_training/dimes_er_small.npz) · [验证记录](../../outputs/v17/dimes_training/validation/run.json)
