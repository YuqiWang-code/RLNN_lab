# v1 MIS 实验

在 RLNN 的动态二值状态搜索中，分别测候选保存、状态分频和动作收益预测。使用本项目生成的 RB200–300 训练图 1,000 张、验证图 100 张，以及 RLD4CO 作者公开测试图 1,000 张。RLNN 各组沿用 FP32、Adam 1e-4、50 epoch、8 图合批、每图状态 batch 64。每图训练轨迹 20×50，推理轨迹 20×100。原 RLNN 的 seed 0 使用已有从头重训的第 43 轮验证最优权重。

| 实验 | 设置 | 种子 |
| --- | --- | --- |
| RLNN | 每条轨迹保留最低能量状态，统一用 A.3 解码后取最大独立集 | 0、1、2 |
| RLNN + Archive | 在 5 个固定步各用 4 个候选的额度，优先选择当前无冲突选中点多的候选，和原能量档案一起按合法大小选优 | 0、1、2 |
| Archive 能量选样 | 同一批时间点、同一附加候选数，按当前能量选额外候选 | 0、1、2 |
| 分频 stats | 原 GCN 加同样的 MIS 状态、度数、已选邻居比例和局部能量收益 | 0、1、2 |
| 分频 dynamic | 惰性随机游走平滑及互补差分，各自线性投影与 SiLU。以 MIS 状态统计为门控，融合后进入同一 GCN | 0、1、2 |
| 分频 low、high、fixed | 分别保留低频、高频，或以 1:1 融合。每组都保留相同的状态统计输入 | 0 |
| RLNN + 成对收益 | 冻结 RLNN，强制一个节点翻转或保持，其他动作与后续 4 步共享随机数。用两次 A.3 解码后的大小差训练轻量 MLP | 0、1、2 |
| RLNN + 一步能量 | 同样的 logit 加权系数，使用精确局部能量收益 | 0、1、2 |
| 分频 + Archive + 成对收益 | 在动态分频权重上重新生成成对监督，推理同时启用三个模块 | 0、1、2 |

状态分频只沿节点维运算。低频是当前状态和邻居均值的平均，高频是原状态减低频，孤立节点的低频等于自身状态。度数取 log1p，已选邻居数与能量收益除以 max(度数, 1)。缓存图拓扑对应的归一化邻接，候选状态的局部统计在每一步重算。[GRACE](https://arxiv.org/abs/2510.12140) · [GCON](https://proceedings.mlr.press/v269/wenkel25a.html)

收益头使用训练集中的 500 图和全验证集。每图在 4 个搜索阶段分别采样 16 个节点动作。目标为强制翻转后继续 4 步的独立集大小减去同一条件下不翻转的大小。用验证 MAE 选头，验证同时记录零预测 MAE、平均配对收益和正收益比例。推理 logit 加上 0.5 倍预测收益。[Rollout](https://arxiv.org/abs/1910.00120) · [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)

## 外部对照

| 方法 | v1 口径 |
| --- | --- |
| Greedy、KaMIS | CPU 单线程。Greedy 每步取剩余子图最小度节点，KaMIS 使用作者 ReduMIS，每图限时 1 秒 |
| RLSA、iSCO | 每图 200 条轨迹、300 步。RLSA 使用上游实现，iSCO 使用 DISCS 的不放回路径提议与反向接受率的 PyTorch 实现，半径固定为 5 |
| GCON | 使用作者 GCON 卷积、局部图结构统计、MIS 损失和 5 个起点的有序贪心解码，同划分重训 50 epoch |
| CAM、DiffUCO | 使用 CAM 作者仓库的 PyTorch 实现和各自训练目标，同划分重训 50 epoch。CAM 每图采样 10×50，状态批 256。DiffUCO 每图采样 20×5，两者测试 50 步 |
| Gurobi、PPO/LwD、INTEL、DGL、DIMES、DIFUSCO、LTFT | 作为论文成绩与源码参考。Gurobi 的通用 MILP 需要能覆盖 RB 图规模的许可证，监督方法需要训练解标签 |

外部基线保留自己的求解与解码规则。RLNN 消融共享 A.3 解码。主表记录每图最好的一个合法解。单图时间覆盖建图后的求解初始化、所有状态更新、候选保存与解码，CUDA 两端同步。加载、图读取、主机到设备传输和独立集复核在计时区间外。显存使用单进程 PyTorch 峰值已分配内存，CPU 方法填 0 MiB。ΔSize 与对应种子的 RLNN 逐图相减，再求平均。种子标准差用各次完整测试的 Mean Size 计算。

固定时间补充比较使用 seed 0 和 0.5 秒/图的共同预算。每次只保留预算内已完成解码的合法解。另取 32 图离线解码全历史，记录原档案和增广档案相对全历史最优的差值。[逐图指标约定](../../docs/temporary/指标.txt)

## 运行

从仓库根目录在服务器运行。环境使用项目已有 rlnn Python，GCON、CAM 需要其固定源码及相应 PyG 依赖。

~~~bash
PYTHON=/home/hzeng/envs/rlnn/bin/python CUDA_VISIBLE_DEVICES=0 bash train_scripts/v1/queue.sh frequency
PYTHON=/home/hzeng/envs/rlnn/bin/python CUDA_VISIBLE_DEVICES=1 bash train_scripts/v1/queue.sh comparison
~~~

独立训练用 train.sh，设置 SEED、RUN_NAME，并用 --frequency 选择 none、stats、low、high、fixed、local 或 dynamic。独立推理用 eval.sh，传入 --save_dir、--label、--result_dir，以及对应的 --frequency、--archive 或 --value_checkpoint。每个训练进程绑定一张卡，独立求解评估由各卡的队列顺序执行。

记录在 outputs/v1/<run>/。per_graph.csv 含图编号、集合大小、求解时间、冲突边数和选中点。运行参数、耗时和显存见 run.json。数据和 checkpoints 留在服务器。工作簿使用 V1汇总、V1逐种子、V1训练和 RLNN论文表1 四个页签。Windows 端可用 python tools/publish_v1.py --watch 按完成的运行同步与发布。
