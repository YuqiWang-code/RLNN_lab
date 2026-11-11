# 静态先验 v2

验证方案见 [RLNN_Static_Prior_v1_Postmortem_and_v2_Plan.md](../../../docs/已归档的想法和文件/mamba/RLNN_Static_Prior_v1_Postmortem_and_v2_Plan.md)。主口径为 FP32 CUDA Graph、K20、验证集前 100 图、每图 3 个搜索种子。P0 默认每图做一轮交错以缩短普查时间。P1 和 P2 保留两轮，第二轮反转图内臂序。预算任务分卡计时。所有臂共用原骨干、A.3 解码和冻结的 PAVE-Stats H1。实验输出位于 `outputs/mamba_v2`。

## P0 接口普查

```bash
bash train_scripts/mamba/v2/check.sh
bash train_scripts/mamba/v2/prepare_oracles.sh --gpu 0 --kamis /path/to/redumis
bash train_scripts/mamba/v2/p0.sh curve --gpu 0 &
bash train_scripts/mamba/v2/p0.sh budget --gpu 1 &
wait
python train_scripts/mamba/v2/summarize.py --ref 0
```

`p0.sh diagnose` 用相同图和搜索种子在 eager 路径重放，报告相同状态上的概率差与翻转差，以及各自轨迹上的翻转和状态分歧。`BACKEND=fp16-eager` 可单独复测，汇总时传入 `--backend fp16-eager`。命令均可通过 `--offset 50 --samples 50` 复核后 50 图，使用 `--output` 指定单独的输出目录，并在 P0 脚本传入对应的 oracle 文件。

P0 的 `amp@0` 同时将常量项和节点项清零，固定步数下 A.3 与 PAVE 的节点集合必须逐位匹配 B。`amp@A` 只注入节点项，保留 v1 的方向和 alpha 符号，A 定义为每图峰值绝对偏置。v1 节点项的原始缩放系数约为 0.019，实际峰值随图变化。`amp@0.02` 提供这一量级的剂量点，`v1-node-only` 复用原始节点偏置，`centered` 复用完整 v1 偏置及其常量项。

每个非零 `amp@A` 配一组 `rand@A`，两臂在同图上的节点偏置 RMS 相同。固定的 `null-rand` 保留每图 RMS 0.0107，用于估计空对照。KaMIS、B 和反向 oracle 都测 0.5、1、2 三档。KaMIS oracle 鼓励加入目标内节点、移除目标外节点，反向 oracle 反转两列。B oracle 用独立多起点搜索中最先达到最大大小的 A.3 可行解。

曲线同时报告各截止步的 A.3 和每 10 步维护的解码归档。到达 T800 大小的步数按已解码的采样点计。固定步数的耗时含曲线采样，部署耗时采用独立的 0.5/1.0 秒协议。后者包含策略准备，给 PAVE 预留 0.05 秒。初始状态机制指标在计时后使用该次求解缓存的偏置计算。离线重放和训练缓存原始结构特征与序，归一化随检查点计算，计时评测每次重新准备。

汇总先在图内聚合种子和重复，再按图配对 bootstrap。null 带取换种子与固定随机臂各自有符号效应点估计的最大绝对值，空对照 CI 和逐图绝对差仍分别报告。P0 用固定步数 A.3 与 PAVE 的配对差判定接口，振幅臂还须优于同 RMS 随机臂。P0 单列两档预算的证据，0.5/1.0 秒同时过线是 P1 准入条件。缺少空对照或配对数据的汇总标为 `incomplete`。

## 条件训练与消融

默认三组 `REF_SEED=0/1/2` 对应训练种子 42/123/999。训练冻结骨干，使用 1,000 张训练图、3 epoch、6,000 次更新、λ=0。K20 中 4 条粒子轨迹用学生策略生成，与同粒子的教师轨迹共享随机流。教师轨迹、实际训练状态与学生列号分别记录。每 epoch 输出 T100/T400/T800 验证结果和机制指标。训练脚本按 `BACKEND` 读取对应评测口径的准入结论。

```bash
bash train_scripts/mamba/v2/train.sh p1 centered8 --gpu 0
bash train_scripts/mamba/v2/train.sh p1 centered5 --gpu 1
bash train_scripts/mamba/v2/train.sh p1 constants --gpu 1
bash train_scripts/mamba/v2/eval.sh p1 curve --gpu 0
bash train_scripts/mamba/v2/eval.sh p1 budget --gpu 1
python train_scripts/mamba/v2/summarize.py --stage p1 --ref 0 \
  outputs/mamba_v2/p1_curve_ref0_fp32-graph outputs/mamba_v2/p1_budget_ref0_fp32-graph
```

P1 的 `masked` 和 `matched-rand` 共用完整模型检查点和常量项，前者关闭节点项，后者以同 RMS 随机方向替换。P1 汇总记录 alpha、门值和实际节点偏置 RMS。各主协议均未过效应门槛且节点偏置 RMS 均低于固定随机对照 RMS 的十分之一时，结论为 `inert`。逐特征消融可向评测脚本传 `--feature-drop core` 等参数和独立的 `--output`。

如果 P0 只通过 oracle，走方向训练。两者都通过时默认先走 P1，汇总同时保留 `oracle_source` 和 `oracle_dose`，也可以独立启动 oracle 分支。

根据 `oracle_source` 生成对应的 1,000 张训练图目标。KaMIS 用 `prepare_oracles.sh --train-targets --kamis /path/to/redumis`，B 用 `prepare_oracles.sh --train-targets --train-oracle b`。训练目标默认写入 `oracles-train_ref<ref>_<backend>`，验证目标写入 `oracles_ref<ref>_<backend>`。将 `ORACLE_TRAIN_TARGETS` 指向生成的 `oracle-kamis-train.json` 或 `oracle-b-train.json`，再运行 `train.sh oracle centered8`。

方向诊断固定 `tanh(alpha)=0.5`，逐图中心化后按绝对峰值归一化。启动脚本将 `static_node_limit` 设为通过剂量的两倍，使有非零方向的图实际注入峰值等于该剂量。oracle 阶段的 `NODE_LIMIT` 覆盖值也按目标注入峰值解释。方向回归权重默认 0.03，可用 `--oracle_weight` 调整。训练日志分别记录 Erdős 损失、方向 MSE 和加权项。该分支可用 `eval.sh oracle curve|budget` 和 `summarize.py --stage p1` 汇总。

P1 通过后，使用通过的 `NODE_LIMIT` 运行 `train.sh p2`，分别指定 `mlp8`、`msg8`、`mamba8`、`mamba8-shuffle`、`mamba8-degree`。这些臂共用 8 维特征、中心化和训练协议。用 `eval.sh p2 curve|budget` 与 `summarize.py --stage p2` 比较结构序 Mamba 和 MLP/MSG。
