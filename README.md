# RLNN Lab

RLNN Lab 基于 Shengyu Feng 和 Yiming Yang 的 RLD4CO 实现，研究最大独立集（MIS）求解器在逐图截止前能交付多大的可行解。粒子搜索在 GPU 上探索二值状态，定时解码出独立集；可行解搜索先在 CPU 上构造独立集，再用学习排序或顶点度指导交换和分支。两条路线分别记录合法节点集的到达时间和每次求解的返回时间。论文把这一求解器族命名为 **SPRINT**：构造路线（先发布可行集再排序）记为 **SPRINT-C**，投机路线（GPU 粒子搜索）记为 **SPRINT-S**，对应 ICML 稿 *SPRINT: Feasible-Set-First Neural Search for Maximum Independent Set Under a 200-Millisecond Deadline*。[RLD4CO 官方仓库](https://github.com/Shengyu-Feng/RLD4CO)

## 共同计时边界下的求解结果

从同一主机边表开始计时，允许三个求解器在搜索中交付可行解。在四档作者图与 50、100、200、500 ms 共 16 个组合上，学习排序版的平均 SIZE 均高于常驻 OnlineMIS。200 ms 时，小 RB、大 RB、小 ER 的学习版和常驻 OnlineMIS 分别为 **20.1150 / 20.0817、42.4417 / 41.7613、44.3438 / 42.7240**，这三档的两臂均按时交付 6,384/6,384 个解。单用顶点度排序在小 RB 也能于 50 ms 内次次最优，在大 RB 四档均取得更高平均 SIZE。小 ER 的同核交错复测中，学习相对度排序的四个图级配对区间均跨零。

在 ER-[9000–11000] 的 16 张作者图上沿用小 ER 的冻结权重与交换搜索配置。200 ms 下，学习排序、度排序、常驻 OnlineMIS 的平均 SIZE 分别为 **370.1250、369.4167、363.5833**，三者均 48/48 次按时交解。按图配对后，学习版比常驻 OnlineMIS 多 6.5417 个顶点（95% CI [5.2917, 8.0417]），比度排序多 0.7083 个（[−0.7500, 2.2500]）。[万节点 ER 逐图统计](outputs/v16/er_large_200ms/summary.csv) · [配对差值](outputs/v16/er_large_200ms/paired.csv)

在 14 张未参与训练或调度选择的寄存器分配干涉图上，三个求解器各自 42/42 次按时返回经证实的全局最优解，平均 SIZE 均为 **208.3571**。两个排序臂在计算优先级之前就已得到最优首解。[完整结果与四预算对照](docs/常驻MIS四预算与迁移.md) · [实验入口](train_scripts/v16/README.md) · [论文 PDF](outputs/paper_20261120/paper.pdf)

主表逐项列出四档作者图的 200 ms 实测 SIZE、实际返回时间和按时交解次数。定性对照固定每档第一张图及全部三个种子，从交付的原始节点集计算各方法与 Fused RLNN 的重合数。 [主表逐项来源](outputs/v16/four_scale_200ms.csv) · [四档定性对照](overleaf/sections/qualitative_cross_scale.tex)

## 三档 200 ms 的重训结果

同一套微调后的 RLNN 节点优先级权重和单线程可行解搜索，在作者 RB-[200–300]、RB-[800–1200]、ER-[700–800] 上依次取得 **20.1150、42.4803、44.3620**，平均返回 **9.81、136.64、197.07 ms**，全部 6,384 次按时有解。小 RB 的 1,000 张图经独立离线核实，三个种子每次均达到全局最优。大 RB 相对调时后的 OnlineMIS 高 0.8663，小 ER 相对 RLSA 高 1.2083。与纯度排序相比，学习排序在两个 RB 档的优势很小或持平，小 ER 的差值为 +0.0807，95% 区间包含零。[三档结果与设计](docs/三档MIS限时评估.md) · [模型权重](outputs/v11/scale_priority_20261121/training/best.pt) · [复跑入口](train_scripts/v15/priority_200ms.sh)

## 排序权重的补充实验

200 ms 下在同一批内比较五个排序臂（仅度数、λ=0.1、λ=0.25、λ=1.0、仅学习分数，同一检查点与调度）。大型 RB 的交付规模落在 42.4387–42.5130，小型 ER 落在 44.1432–44.4245，小型 RB 五臂同为 20.1150（全部已证明最优）。相对同批度数臂的差值：大型 RB 为 +0.0187 至 +0.0743，小型 ER 为 −0.1068 至 +0.1745，大型 ER 的区间都含零。实际使用的 λ=0.25 在小型 ER 上最好；大型 RB 上更大的权重还能再多约 0.02 个顶点。混合排序与 −度数 的 Spearman 相关为 0.9903、0.9906、0.9986，学习分数本身为 0.5829、0.6451、0.9562，说明学习项主要在度数排名接近的顶点之间重排。[逐图记录与统计](outputs/v17/priority_mix_20261122/) · [运行说明](outputs/v17/priority_mix_20261122/README.md)

## 原融合引擎的实验结论

在 RB-[200–300] 的严格 200 ms 同图同种子对照中，融合 RLNN 平均 SIZE 为 **20.0300**，相对发布版 RLNN 高 **0.6053**，相对 RLSA 高 **0.0793**。另用三个新种子复测融合版和单线程 KaMIS OnlineMIS：合并六种子后，两者为 **20.0273 / 20.0365**，按时有解 **6000/6000 / 5995/6000**。 [论文 PDF](outputs/paper_20261120/paper.pdf) · [Overleaf 源码包](outputs/paper_20261120/overleaf-source.zip) · [实验与配对统计](docs/论文实验.md) · [图表设计参考](docs/论文图表参考.md)

沿用同一组冻结权重与 200 个粒子，融合 RLNN 在 RB-[800–1200] 上取得 **28.9847（2613/3000 按时）**，在 ER-[700–800] 上取得 **40.8854（384/384）**。在 ER-[9000–11000] 上，装载单图最少需要 1.84 秒，48 次都在搜索前用完预算。每档的 SIZE 都包含未按时交解的零分。 [四档定量表](overleaf/sections/scale_table.tex) · [完整数据](outputs/v11/scale_deadline_20261120/comparison.csv) · [装载与解码耗时](outputs/v11/scale_deadline_20261120/fused_timing.csv)

同一批 1000 张 RB-[200–300] 作者测试图、同一台 RTX 5090 上，**融合 RLNN（关闭 PAVE）达到 20.0300 @ 193.80 ms/图**。作者公开 RLNN 的复测为 **19.520 @ 208.07 ms/图**，固定 300 步 RLSA 复测为 **19.973 @ 208.39 ms/图**。相对 RLSA 的逐图配对 SIZE 增益为 **+0.057**，95% CI **[+0.040, +0.074]**。RLSA 比较对应近似等墙钟工作点。 [论文证据与消融计划](docs/论文实验.md) · [RLNN 配对](outputs/v11/fused_pave_02s_deadline/vs_rlnn.json) · [RLSA 配对](outputs/v11/fused_pave_02s_deadline/vs_rlsa.json)

### 0.2 秒融合引擎（2026-11-18）

在同一张 RTX 5090 上，对 1000 张 RB200–300 作者测试图和搜索种子 1017、1019、1021 逐图配对。两臂均使用冻结 RLNN、`K=200`、dense FP16 和 `torch.compile`，关闭前导解码，A.3 解码取最低能量的 4 个粒子。PAVE 臂预留 40 ms，关闭 PAVE 的一臂将该预算留给搜索与解码。

| PAVE | 搜索解均值 | 最终解均值 | 平均搜索步数 | 平均墙钟 | 200 ms 内有解 | 墙钟超时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 开 | 20.0113 | 20.0147 | 278.4 | 185.75 ms | 3000/3000 | 9/3000 |
| 关 | 20.0300 | **20.0300** | 351.0 | 193.80 ms | 3000/3000 | 0/3000 |

关闭 PAVE 后，最终解平均增加 **0.0153**（图级配对 bootstrap 95% CI [+0.0097, +0.0213]，55 胜 / 2936 平 / 9 负）。PAVE 在 10/3000 次求解中改进了搜索解，平均 +0.0033。这个预算下，把预留时间交给融合搜索得到更大的最终解。

逐图时钟包含引擎装载、搜索、中段与末次 A.3 解码，以及启用时的 PAVE。求解器为末次解码预留至少 5 ms，并保留搜索窗口中段已完成的合法解。超过截止时刻的更新不计分，原始墙钟超时另行记录。两臂的末次解码均在截止前完成。PAVE 臂的 9 次墙钟超时均沿用截止前的搜索解，墙钟最长 202.0 ms。一次性编译、CUDA Graph 捕获、数据读取和公共图预处理在逐图时钟外。

RLSA 的近似等墙钟参照为 **19.9620 @ 201.0 ms/图**，来自同一作者集上的固定步数运行。 [RLSA 时钟扫描](docs/temporary/RLNN_v8_SOTA_Direction.md) · [配对统计](outputs/v11/fused_pave_02s_deadline/summary.json) · [逐图结果](outputs/v11/fused_pave_02s_deadline/) · [复跑脚本](train_scripts/v11/fused_pave_02s.sh)

### 同一策略权重下的收益

固定作者权重和 20 个粒子，切换求解流程后 SIZE 从 19.4247 升至 19.9773；继续使用作者权重，将并行粒子增至 200 个后达到 20.0360。相同融合配置下，本地权重与作者权重的配对差为 −0.0060，95% CI [−0.0150, +0.0033]。首项包含解除 100 步上限和更换搜索、解码实现。 [权重与粒子数配对](outputs/v11/paper_ablations/checkpoint_bridge_20261120/summary.json)

### CUDA Graph × 编译正式消融

固定 130 步时，两项都开启的平均墙钟为 74.53 ms/图，两项都关闭为 277.70 ms/图。200 ms 截止下，两项都开启完成平均 348.6 步，实现 **20.0287 @ 193.57 ms/图**，3000/3000 次按时交付。相对两项都关闭，配对 SIZE 增加 **0.3320**，95% CI **[0.3093, 0.3563]**。 [四臂结果与协议](docs/论文实验.md#cuda-graph--编译) · [配对统计](outputs/v11/paper_ablations/cuda_graph_compile_20261119/summary.json)

### 1 秒参照

融合、关闭 PAVE 的 1 秒墙钟实验三种子均值为 **20.0870**，平均 1852.4 步。RLSA 的 1 秒插值约为 20.084，两者相差 +0.003。v11 A0 的 **20.0690** 对应 seed 1017 的固定 810 步复现协议。 [融合消融](docs/temporary/RLNN_v13_Ablation_Results.md) · [A0 三种子结果](outputs/v11/t7/summary.md)

## 实验与交接

[项目现状](docs/项目现状.md) 汇总各阶段结论、对外比较口径和未决事项。 [实验输出](outputs/README.md) · [正式结果](docs/实验结果.xlsx) · [冒烟结果](docs/冒烟结果.xlsx) · [v8–v13 文档](docs/temporary/README.md) · [历史归档](docs/已归档的想法和文件/README.md)

根目录 [overleaf](overleaf/README.md) 提供 ICML 2027 匿名论文工程。正文区分粒子解码与可行解搜索，比较二者在截止前交付的节点集。讨论与附录将同一选择问题映射到冷链派车。[双路线模型图](overleaf/figures/search_paths.pdf) · [可编辑模型图（第 5 页）](docs/模型图.pptx) · [常驻求解器曲线](overleaf/figures/resident_deadlines.pdf) · [解码轨迹](overleaf/figures/decode_cases.pdf) · [冷链算例](overleaf/figures/dispatch_example.pdf)

## RLNN复现基线

从随机初始化训练 50 轮，共 100,000 次 Adam 更新。验证集选出的第 43 轮权重在作者公开测试集上取得 **SIZE 19.549**，比论文表 1 的 19.52 高 0.029。[实验报告](docs/实验报告.md) · [论文表 1](https://arxiv.org/pdf/2502.00277#page=7)

| 权重 | 显卡 | SIZE ↑ | TIME ↓（分钟／500 图） |
| --- | --- | ---: | ---: |
| 从头重训 · 第 43 轮 | RTX 5090 | **19.549** | 4.89 |
| 旧基线 · 第 48 轮 | RTX 5090 | 17.792 | 2.55 |
| 作者公开 erdoes | RTX 5090 | 19.520 | 1.73 |
| 论文表 1 | RTX A6000 | 19.52 | 1.64 |

前三行均使用作者公开的两组各 500 张图和附录 A.3 梯度翻转解码。SIZE 为每图平均独立集大小。TIME 为两组逐图采样与解码总耗时的平均值，各次运行独立计时。[实验结果工作簿](docs/实验结果.xlsx) · [逐图测试记录](outputs/v000_rlnn_baseline/rb_train_author_b64_g8_gpu1_seed0_a3/per_graph.csv)

## 在服务器复跑限时对照

从项目根目录运行 0.2 秒融合引擎的 PAVE 配对实验。命令使用作者测试集张量缓存、冻结 v2 RLNN 和 v3 PAVE-Stats H1 权重。末尾的 `1` 为 GPU 编号。

```bash
RESULT_DIR="outputs/v11/fused_pave_02s_$(date +%Y%m%d_%H%M%S)" \
  bash train_scripts/v11/fused_pave_02s.sh 1
```

历史 v5 的 0.5 秒对照使用验证张量缓存、种子 0 的 v2 RLNN 骨干和 v3 PAVE-Stats H1 权重。

```bash
PYTHON="$HOME/envs/rlnn/bin/python" CUDA_VISIBLE_DEVICES=0 SEED=0 \
  SPLIT=validation SAMPLES=100 BUDGET=0.5 POST=pave \
  RUN_NAME="readme_budget_seed0_$(date +%Y%m%d_%H%M%S)" \
  bash train_scripts/legacy/v5_frequency/eval_budget.sh baseline --extended-search
```

结果写入 `outputs/v5_frequency/<RUN_NAME>/pave/`。完整训练、诊断和双档预算矩阵见 [v5 运行说明](train_scripts/legacy/v5_frequency/README.md)。

## 从头复跑 RLNN 基线

从项目根目录运行。已有训练缓存和作者测试图的部署可直接运行重训入口。

新部署先准备环境和本地生成的 1,000 张训练图、100 张验证图，再生成张量缓存。

```bash
python3.10 -m venv "$HOME/envs/rlnn"
source "$HOME/envs/rlnn/bin/activate"
bash train_scripts/legacy/v000_rlnn_baseline/setup_env.sh
bash train_scripts/legacy/v000_rlnn_baseline/prepare_data.sh
bash train_scripts/legacy/v000_rlnn_baseline/prepare_cache.sh
```

将作者发布的 MIS RB200–300 测试文件 `idx_0.gpickle` 至 `idx_999.gpickle` 放入 `models/data/author_rld4co/MIS/RB_small_test/`。评估按编号分成 0–499 和 500–999 两组。[作者测试图](https://huggingface.co/datasets/shengyuf/RLD4CO/tree/f273683b3fe6faa217de35ead96fbe0ee4a44a4a/MIS/RB_small_test) · [数据集说明](docs/数据集.md)

```bash
PYTHON="$HOME/envs/rlnn/bin/python" bash train_scripts/legacy/v000_rlnn_baseline/retrain_author_erdoes.sh
```

脚本在 GPU 1 上每步合批 8 张图，每图最多 64 个状态，以单卡近似作者公开脚本的 8 卡更新方式，每轮更新 2,000 次。训练结束后，脚本用验证最优权重自动跑两组 A.3 测试。本次采用公开脚本的 batch 64，论文附录 A.2 记录的是每卡 32。[训练配置对照](docs/实验报告.md#按作者脚本从头训练) · [服务器说明](docs/服务器.md)

## 目录与结果整理

| 入口 | 内容 |
| --- | --- |
| [models/](models/README.md) | 骨干、A.3 解码、PAVE、CUDA Graph 搜索与归档源码 |
| [train_scripts/v11/](train_scripts/v11/README.md) | 现行限时对照、A0、v11 诊断与融合消融 |
| [outputs/](outputs/README.md) | 正式运行、逐图记录和统计来源 |
| [数据集](docs/数据集.md) · [服务器](docs/服务器.md) | 数据划分、部署环境和 checkpoint 位置 |
| [tools/](tools/README.md) | 运行记录、同步、结果工作簿与快照 |

权重及完整图数据保存在服务器。正式和冒烟指标分别记录在两本工作簿。

## 参考文献

Shengyu Feng 和 Yiming Yang，*Regularized Langevin Dynamics for Combinatorial Optimization*，ICML 2025，PMLR 267，16977–16992。[正式论文页](https://proceedings.mlr.press/v267/feng25r.html) · [本地 PDF](docs/参考文献/神经组合优化/RLNN%28ICML2025%29.pdf) · [文献索引](docs/参考文献/文献索引.md) · [源码来源](docs/源码来源.md)

结构补偿一轮另整理了 2024–2026 年的 17 篇参考文献（13 篇 CCF-A）：度数偏误与高度数子群、过平滑/过挤压的残差补救、非消息传递特征通道与门控、图组合优化中的局部信息建模。PDF 按方向收在 `docs/参考文献/结构补偿/`，作者代码快照见 [参考实现](reference_methods/README.md)。[S01–S17 索引](docs/参考文献/文献索引.md#结构补偿2026-11-11)
