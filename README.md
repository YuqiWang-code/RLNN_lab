# RLNN Lab

RLNN Lab 基于 Shengyu Feng 和 Yiming Yang 的 RLD4CO 实现，研究 RB200–300 图上的最大独立集（MIS）。RLNN 负责搜索初始解，PAVE-Stats H1 在合法解上做平台搜索。[RLD4CO 官方仓库](https://github.com/Shengyu-Feng/RLD4CO)

## 当前方法与实验结论

**论文主张（固定，2026-11-13）。** 在 RB200–300 上，把冻结的神经搜索循环做成图捕获块、再挂一个学习型后处理平台，可以在 1 秒内达到 19.9+，在 SIZE 与 TIME 两轴上同时超过 KLSA 的 35 秒 19.97 与 RLNN 的 98 秒 19.52；两个组件的作用可分离：捕获贡献 +0.28（同钟步数 ×10.5），平台贡献 +0.22，学习排序器相对规则/随机再加 +0.16。

这是 2026-11-13 固定的**原始主张原文**（不改写，留作对照）。它测完了：前两句部分成立、后两句不成立，逐条判定见下方「实测」。**对外写作只能用实测口径，不能引用原文的最后两个分句。**

> 命名更正（2026-11-14）：原文的「KLSA」在论文表里是 **RLSA**。RLNN 与 RLSA 是同一篇论文（Feng & Yang, ICML 2025, *Regularized Langevin Dynamics for CO*）在 RLD 框架上的两个求解器——**RLSA** = SA 版（TYPE `H`，纯启发式，无神经网络，RB[200–300] 19.97@35 s 的持有者），**RLNN** = 神经网络版（TYPE `UL`，无监督局部目标训练，19.52@1.64 m）。所以「压过 RLNN」压的是他们的 NN 变体；19.97 那格来自他们的 SA 变体。

这条主张的每一格都必须满足三个前提，缺一格就整条撤回：

1. 在 `RB_small_author_test` 的 1000 张作者测试图上重测，不能只报 100 张验证图；
2. 三轴（SIZE、TIME、CI）与 RLSA / RLNN / KaMIS 同卡、同图、同时间网格重测，不能拿别人的 A100 墙钟对自己的 5090 墙钟；
3. 两个组件各自可消融、且叠加后仍然成立（2×3 六格：`{eager, captured} × {无 PAVE, PAVE-规则, PAVE-学习}`）。

**实测（2026-11-13，1000 张作者测试图，同卡、同一 1 秒网格）。** 三档都跑完后的账面：

| 时钟 | 格子 | 裸搜索 | +平台 | 平台增益 | 步数 |
| --- | --- | --- | --- | --- | --- |
| 0.5 s | eager/eager | 18.7400 | 19.3780 | +0.6380 | 49 |
| 0.5 s | captured/cached | 19.8243 | 19.8403 | +0.0160 | 402 |
| 1.0 s | eager/eager | 19.7317 | 19.7717 | +0.0400 | 213 |
| 1.0 s | captured/cached | 19.9293 | **19.9333** | +0.0040 | 1058 |

对照同卡实测：KaMIS 1 s **20.1060**（5 s 起饱和在 20.05）、RLSA 19.97@35 s、RLNN 19.52@1.64 m【论文】。

**口径更正（2026-11-14，重要，撤回此前的时间比较）。** 论文 Table 1 的 TIME 是**整个测试集**的总时间、逐图串行（p.8），而 RB-[200–300] 的测试集是 **500 张**（p.6）。所以 RLNN 1.64 m = **0.197 s/图**、RLSA 35 s = **0.070 s/图**、KaMIS 1.40 h = 10.1 s/图。我们的 1 秒是**每图**预算 → 折成同一口径 500 s/500 图，对 RLNN **慢 5.1 倍**、对 RLSA **慢 14 倍**。此前写的「快 98 倍 / 快 35 倍」把每图预算与整集合计混算，已作废。

逐条：

- 「1 秒内 19.9+」**成立**：19.9333。
- 「超过 RLNN 19.52」**只有 SIZE 轴成立**（+0.41）；TIME 轴按同一口径我们慢 5.1 倍，此前「快 98 倍」作废。0.2 s/图的等钟补测在跑（`author_short.sh`），若那一档 ≥19.52 才能改写回双轴。
- 「超过 RLSA 19.97」**两轴都不成立**：SIZE −0.037；TIME 按同口径慢 14 倍。原主张这一句只能改成「SIZE 与 RLSA 同水平（−0.037）」。
- 「两组件可分离相加」**不成立**：1000 图上 capture on search 在 0.5 s 是 **+1.0843** [+1.0300,+1.1387]、1.0 s 是 **+0.1977**；platform on eager 是 +0.6380 / +0.0400，但 platform on captured 只剩 **+0.0160 / +0.0040**。步数一多 PAVE 就无事可做，**捕获与平台是替代关系**；原主张的「+0.28 / +0.22 / +0.16」三分法来自不同协议，不能相加。
- 「19.5 以内的 0.5 秒档」不成立：0.5 s 最好 19.8403。

详细表与复现见 [作者集主表](docs/temporary/RLNN_v8_Paper_Table.md)。

**限时部署采用原骨干持续搜索＋冻结 PAVE-Stats H1。** PAVE 的后处理收益来自 v3 的 1,000 张作者测试图。骨干部署选择来自 v5 的 100 张验证图、三种子预算对照，两组结果各按原协议报告。[v3 结果与口径](docs/temporary/RLNN_v3_Results_Analysis.md) · [v5 部署对照](docs/已归档的想法和文件/RLNN_v5_Frequency_Results.md)

| 版本 | 结论 | 结果与入口 |
| --- | --- | --- |
| v3 · PAVE-Stats H1 | 同一批合法起点上，固定工作量的 Mean Size 为 19.6633，相比 rule23 提高 0.0603。 | [分析](docs/temporary/RLNN_v3_Results_Analysis.md) · [运行协议](train_scripts/legacy/v3/README.md) |
| v4 · 双尺度频域适配器 | 短筛未显示频域独立收益。机制追诊把高支路的驻留分量与动作收益分开验证。 | [短筛结果](docs/已归档的想法和文件/RLNN_v4_Frequency_Results_Analysis.md) · [机制追诊](docs/已归档的想法和文件/RLNN_v4_Frequency_Mechanism_Followup_Results.md) · [训练与评测](train_scripts/legacy/v4_frequency/README.md) · [追诊复算](train_scripts/legacy/v4_frequency/followup/README.md) |
| v5 · DAFR-Lite | 112 参数的输出残差达到高支路零驻留目标。FP32 固定 100 步的 Mean Size 为 19.580，MSG 为 19.583，MLP 为 19.587。两档预算也未通过准入。 | [实验报告](docs/已归档的想法和文件/RLNN_v5_Frequency_Results.md) · [阶段决策](outputs/v5_frequency/analysis/decision.json) · [训练与评测](train_scripts/legacy/v5_frequency/README.md) |
| v6 · SC-RLNN | 三种子首轮筛选中，固定 100 步＋PAVE 的 Mean Size 为 19.073，配对 B 为 19.657。速度和质量均未通过准入。 | [模型与首轮结果](docs/已归档的想法和文件/RLNN_v6_SC_RLNN.md) · [训练与评测](train_scripts/legacy/v6/README.md) |
| Mamba v0 · PC-OSM-RLNN | 一次性扫描正确。宽度 16 在 FP32/FP16 下仍为 B 的 1.216×/1.249×，同信息代理对照未显示稳定增量，停在 P0。 | [P0 报告](docs/已归档的想法和文件/mamba/RLNN_Mamba_v0_P0_Results.md) · [运行入口](train_scripts/mamba/v0/README.md) |
| 静态先验 v1 | 中心化 MLP 的 T100 耗时为 B 的 1.041×，0.5 秒两轮均为 20.12，B 为 20.14。与同权重去节点项的版本在 T1000 和两档预算的 250 个配对上大小相同。 | [v1 结果](docs/已归档的想法和文件/mamba/RLNN_Static_Prior_v1_Results.md) · [运行入口](train_scripts/mamba/v1/README.md) |
| **Mamba 线（已归档）** | v2 的 P0 证明注入接口有杠杆（oracle T100 +0.45、0.5 秒 +0.19、反向 −0.29），但 P1 的三次方向学习都失败：搜索目标 −0.04、权重主导 +0.05、纯监督 0.088（留出 AUC 0.6375）。**瓶颈在 8 维静态特征的输入信息**，T100 实测 +0.003。 | [归档总结](docs/已归档的想法和文件/mamba/README.md) · [定价曲线](docs/已归档的想法和文件/mamba/RLNN_Mamba_v2_P0_Results.md) |
| 结构补偿 v1（已归档） | 层内注入度数残差/度数门控/显式增益三个零初始化通道。`degree` 在 T100＋PAVE 得 **+0.090**（CI [−0.020, +0.200]，19/70/11），是首个非零信号；`gate` +0.060、`gain` +0.030、`all` −0.020。速度门失败（1.11×–1.29×，基线是发射受限）。**未过准入**。 | [结果](docs/已归档的想法和文件/structure/RLNN_Structure_v1_Results.md) · [设计](docs/已归档的想法和文件/structure/RLNN_Structure_v1_Design.md) |
| 结构补偿 v2（已归档） | 把度数通道改到读出端（一次广播加，静态量按图缓存），增益改可学正尺度，每图 3 个搜索种子。`degree_out` 只有 **+0.013**（CI [−0.040, +0.063]，20/64/16），`all_out` 与它逐位相同，速度仍 1.11×。**v1 的 +0.090 未复现，判为单种子波动**。 | [结果](docs/已归档的想法和文件/structure/RLNN_Structure_v2_Results.md) · [归档总结](docs/已归档的想法和文件/structure/README.md) · [汇总](outputs/structure_v2/analysis/summary_ref0.json) |
| v7 · TCCR（已结束） | 不改翻转概率，改用轨迹信息。固定工作量下，K=20 合法解的**精确重组**在冻结 PAVE 之上 **+0.088**（三种子 69 胜 231 平 0 负）；冲突释放区域 `|U| ≤ 4` 只有 +0.043（8–15/100 图），预注册判决 `stop`。**预算门否掉整条线**：两个参考种子、两档预算共四格全部 CI 跨零（0.5 秒 −0.028／+0.030，1.0 秒 +0.010／+0.003），20 ms 的搜索价值与重组收益同量级、符号随争用翻转。 | [结果与判决](docs/temporary/RLNN_v7_TCCR_Results.md) · [普查预注册](docs/temporary/RLNN_v7_TCCR_Prereg.md) · [预算门预注册](docs/temporary/RLNN_v7_TCCR_Prereg_Budget.md) · [运行入口](train_scripts/v7/README.md) |
| v8 · 宏步 ACMD（P0 停止） | 把一个 h=2 宏步当成一个联合分布来建模（读出 1→2M、混合分量、末态+记录态双头）。先跑两个只用冻结模型的 P0 诊断：**打乱跨节点相关性**只让中间能量差 1.1–1.4、冲突多 0.5–0.8，最终解大小差 −0.008 到 −0.031（四格 CI 全跨零，方向为 joint 略差）；**只保留端点档案**的最终解差 +0.017（CI [−0.033, +0.083]，仅 3/60 图）。按预注册判读表"两项均无最终质量信号"，**不进入成本门与训练**。| [P0 结果与决定](docs/temporary/RLNN_v8_P0_Results.md) · [方案与判读](docs/temporary/RLNN_lab_v8_review_and_research.md) · [运行入口](train_scripts/v8_macro/README.md) |
| **v8 · 定位与步数（新）** | 冻结策略每步只花 0.26 ms 算术，其余是发射延迟；把 10 步捕获成 CUDA 图块后，**同样的墙钟多 6.75×/7.88× 步数**，0.5 秒 **19.690 → 19.885**（+0.195 [+0.125,+0.270]，21/79/0），1.0 秒 **19.775 → 20.000**（+0.225 [+0.140,+0.320]，22/78/0），闸门先断言轨迹逐位一致。再把 A.3 解码搬进图块（+0.045/+0.010）并跳过重复解码（+0.000/+0.010），合计 **+0.280 [+0.195,+0.370] / +0.235 [+0.150,+0.330]**、30/70/0 与 22/78/0、步数 ×10.5/×11.8——**700 步之后每翻倍只值 +0.01，速度工程到此为止**。等钟定位：KaMIS 的墙钟地板是 1.87 s/图（即使传 `--time_limit=0.1`），所以 **0.5–1.8 秒只有本方法能交解**；在 ≥1.9 秒的四个等钟格上配对差全跨零、93–96/100 图相同，**两者同一水平**。 | [定位与步数结果](docs/temporary/RLNN_v8_Positioning_Results.md) · [汇总表](outputs/v8_macro/positioning/summary/positioning_ref0.md) · [运行入口](train_scripts/v8_macro/README.md) |
| **v8 · PAVE 外接（否）** | PAVE 是否 solver-agnostic？把 KaMIS 0.1 秒的解喂进 A.3＋PAVE：100 张图**每一项增益恰好为 0**（0/100/0，±0.0000，CI [0,0]），三个选择器（学习/规则/随机）全部零交换、再给 0.1 秒 KaMIS 也是 0——KaMIS 的解已是该邻域的不动点。正对照（同一套代码、贪心最小度起点）PAVE 拿到 **+0.710 [+0.560,+0.870]**，其中 +0.55 来自「做局部搜索」本身，学习选择器再加 **+0.160 [+0.080,+0.250]**（15/84/1）。结论：PAVE 的增益随输入收敛程度单调下降（+0.710 / +0.22 / +0.000），**不能写成「接在任何求解器上都能提升」**。 | [外接判决结果](docs/temporary/RLNN_v8_PAVE_External_Results.md) · [运行入口](train_scripts/v8_macro/README.md) |
| **v8 · 跨族迁移（否）** | 冻结策略在没见过的图族/规模上还成立吗？四族各 40 图、1.0 秒等钟、同一冻结骨干＋PAVE：同分布对照 −0.075、ER **−0.625 [−0.900,−0.325]**、BA **−1.775 [−2.200,−1.400]**、n≈1010 的 RB **−3.650 [−4.025,−3.275]**（相对 −0.4 %/−1.4 %/−1.6 %/−8.6 %，随离训练分布的距离单调）。更难看的是等钟下**贪心最小度 + PAVE 直接在 BA（25/40 图）与 RB_large（PAVE 给满 1 秒时 18/40 图）上赢过冻结策略**。反过来 PAVE 的修复与族无关（+0.55/+1.58/+1.58），只要它的时钟随 n 缩放（0.1 秒→1.0 秒）。**迁移不成立，退化是族与规模两条同时出现**。 | [判决结果](docs/temporary/RLNN_v8_Regime_Probe.md) · [汇总表](outputs/v8_macro/regime/regime/summary_ref0.json) · [运行入口](train_scripts/v8_macro/README.md) |
| **v8 · 作者集主表（现行）** | 论文主张在 1000 张作者测试图上测完。1.0 秒 captured/cached **19.9333**（裸搜索 19.9293，平台只剩 +0.0040 可做），对未编译的 19.7717 是 **+0.1617 [+0.1457,+0.1780]**（345/651/4）；0.5 秒 19.8403。同卡对照：KaMIS 1 秒 **20.1060**（5 秒即饱和在 20.04）、RLSA 19.97@35 s、RLNN 19.52@98 s。判定：**「1 秒内 19.9+」成立、「双轴压过 RLNN」成立、「SIZE 压过 RLSA」不成立（−0.037）**；2×2 分解显示平台只在慢搜索上值钱（eager 臂 +0.6380，捕获臂 +0.0160），**两组件是替代关系而非相加**。ER-[700–800]（p 用 KaMIS 反标定为 0.15，128 图）我们 1 秒 42.18，输 KaMIS 44.00 与 RLNN 43.34，**按 scope 处理**。 | [作者集主表](docs/temporary/RLNN_v8_Paper_Table.md) · [CSV](outputs/v8_macro/author_test/capture_budget/capture_budget_ref0.csv) · [运行入口](train_scripts/v8_macro/README.md) |
| **v8 · 捕获摊薄（现行）** | 把「每图重捕获搜索块」（93.5 ms）改成**整个数据集只捕获一次**（`models/mis_v8_pad.py`：缓冲按 `(n_max, capacity)` 固定，每图只改写边缓冲，prelude 6 ms）。1000 张作者测试图、单种子：0.05 秒 **19.4040**（67 步）、0.1 秒 **19.6890**（178）、**0.2 秒 19.8410**（414）、0.5 秒 **19.9450**（1310）、1.0 秒 **19.9830**（2887）。同钟对比：**0.2 秒对我们 19.841 vs RLNN 19.52@0.197 秒 → 两轴同时赢 RLNN（+0.32）第一次成立**；但对 RLSA 19.97@0.070 秒仍差 0.42（要到 1.0 秒才越过 19.97，慢 14 倍），对 KaMIS 20.106@1.0 秒仍低 0.12。 | [摊薄实验](docs/temporary/RLNN_v8_Amortised_Capture.md) · [CSV](outputs/v8_macro/padded_short/padded/padded_ref0.csv) · [运行入口](train_scripts/v8_macro/README.md) |

两条已关闭的线留下一个共同结论：**在 RB200–300 上，无论把静态结构信息放在输出端（Mamba 线）还是放进骨干内部（结构补偿：v6 换骨干、v1 层内注入、v2 读出端融合三种形态），对同预算解大小都没有可复现的增量**。因此 v7 不再改翻转概率的编码方式，转向**利用已经算出来的轨迹信息**：K=20 个 A.3 合法解的精确二部图重组，在冻结 PAVE 之上再得 **+0.088**（三种子 69 胜 231 平 0 负），而冲突释放区域没有可用的量。部署方案是 B 持续搜索 ＋ 精确重组 ＋ 冻结 PAVE-Stats H1（重组为待验证的候选，预算档未过）。

v5 验证中，原骨干持续搜索＋PAVE 在 0.5 秒和 1.0 秒预算下分别取得 **19.677** 和 **19.830**，DAFR＋PAVE 分别为 19.500 和 19.617。[逐种子与配对汇总](outputs/v5_frequency/analysis/) · [实验结果工作簿](docs/实验结果.xlsx)

实验使用两张 RTX 5090。各任务独立单卡运行，训练可同卡并行，正式计时每卡运行一个评测进程。

## 下一步路线（固定，2026-11-14）

现在卡在 **19.98 @ 1 秒**（KaMIS 20.106）。四条独立证据说明这不是「还没跑够」，而是**移动集太小**：

| 尝试（都把答案打在 19.98 之上） | 1 秒档收益 |
| --- | --- |
| 末端 PAVE（修 + 换） | +0.004 |
| König 精确重组（K=20 个合法解二部图） | +0.024 |
| 把交换插进搜索循环（每 60 步一次） | ≤ 0（0.2 秒档 −0.100） |
| 随机踢动（删 5% 再补） | −0.033 |

即：20 个粒子的档案并集已经是 **1-flip 与 (2,1)-swap 双邻域都空**的局部最优。所以路线是「**先给网络全局视野，再把动作换成更大的移动**」，按顺序做，每步都有明确的死刑判据：

1. **全局 token（半天，先做）.** `models/mis_v8_global.py`：每层把 `mean/max` 池化的图级向量投影回隐藏宽度，广播加到每个节点上；只改 forward，损失、步数、数据全不变。训练入口 `train_scripts/v8_macro/global_train.sh`（`--arch global`，50 epochs、`num_tp=50`，与冻结基线同配方）。**判据：1.0 秒 ≥ 20.00（冻结捕获 19.9333 + 0.05）才留下**，否则说明瓶颈确实不在感受野，直接进第 2 步。注：池化按真实节点算，所以这一臂用逐图捕获引擎评测（摊薄的 padding 会污染均值）。

**已完成，结果中性（判据未过）**：热启动微调 6 epoch 后，500 张作者图上 1.0 秒 19.9920 对冻结 19.9900（配对 +0.0020 [−0.0220, +0.0280] 22/457/21）、0.5 秒 −0.0180 [−0.0480, +0.0120] 25/441/34；验证目标与冻结基线持平。详见 `docs/temporary/RLNN_v8_Global_Token.md`。注意冻结骨干当年是**热启动**训练的（`run.json` 里有 `warm_start`），从零训练不可直接比较。

2. **学习型交换头（2–3 天，主线）.** 把动作从「翻一个点」换成「学出来的交换对」：`add(u)=MLP(h_u)`、`del(v)=MLP(h_v)`，配对得分 `s(u,v)=add(u)+del(v)+φ([h_u,h_v,h_u−h_v])`，用**最大权匹配**（复用 `mis_v7_recombine` 的 König 实现）挑一组互不冲突的对，一次性加 A 删 R（允许 |A|>|R|，即 (2,1)/(3,2) 交换）。每 N 步做一次交换、其余步仍 1-flip，训练信号沿用现成的一步目标（把 flip 换成 pair 的 softmax）。**判据：1000 张作者图、1.0 秒 ≥ 20.05 才算成功**；20.05 以下一律视为失败（写进负结果）。

3. **学习式退火（2 天，仅在第 2 步失败时做）.** 加一个价值头 `V(s)` 预测这条轨迹最终的档案大小（标签来自冻结搜索自己的 rollout，不需要 RL），把接受准则从「能量下降才翻」换成学出来的 Metropolis（`ΔV > −τ(t)·log u`，τ 退火）。依据很硬：**同一篇论文里纯 SA 的 RLSA（19.97@0.07 秒）比他们自己的神经网络版 RLNN（19.52@0.197 秒）强 0.45 且快 2.8 倍** —— 「贪心接受」本身就是短板。

三步全失败时，按 `docs/temporary/v8_paper_snapshot_A.txt` 里的「版本 A」口径写论文（三段主张：摊薄捕获赢 RLNN、平台增益是输入次优程度的函数、九条方法线的负结果），不再改网络。

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

从项目根目录执行。已有验证张量缓存、种子 0 的 v2 RLNN 骨干和 v3 PAVE-Stats H1 权重时，可复跑 0.5 秒的持续搜索对照。

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

| 目录 | 用途 |
| --- | --- |
| `models/` | 上游源码与模型迭代 |
| `train_scripts/legacy/v000_rlnn_baseline/` | 环境、数据、训练、评估和冒烟入口 |
| `outputs/v000_rlnn_baseline/<run>/` | 日志、配置和数值结果，模型权重保存在服务器 |
| `train_scripts/legacy/v3/` | PAVE 起点生成、标签、训练和评测 |
| `train_scripts/legacy/v4_frequency/` | 共享教师状态训练、独占计时验证、频谱与收益诊断 |
| `train_scripts/legacy/v5_frequency/` | DAFR-Lite 短训、固定步数与限时评测、机制诊断 |
| `train_scripts/legacy/v6/` | SC-RLNN 训练、速度准入、固定步数评测与逐层诊断 |
| `train_scripts/mamba/` | 已冻结的 Mamba / 一次性静态先验线（v0 预检、v1 静态先验、v2 接口普查与方向判决） |
| `train_scripts/structure/` | 已冻结的结构补偿 v1/v2 线：模块自检、训练、固定步数与限时评测、种子池化汇总与准入 |
| `train_scripts/v7/` | v7 TCCR 普查：算子自检、K=20 捕获、重组/区域普查、端到端重组+PAVE、门槛汇总 |
| `train_scripts/v8_macro/` | v8：内核自检、P0 诊断、定位与解码、PAVE 外接、跨族迁移、作者集主表（P0–P18 的完整命令链，含分片并行） |
| `outputs/v3/`、`outputs/v4_frequency/`、`outputs/v5_frequency/`、`outputs/v6/` | 各版本逐图记录、配对汇总与阶段决策 |
| `outputs/v7_context/`、`outputs/static_prior_v1/`、`outputs/mamba_v2/` | Mamba 线的预检、静态先验与接口普查证据 |
| `outputs/structure_v1/`、`outputs/structure_v2/` | 结构补偿的逐图记录、配对汇总与准入判决 |
| `outputs/v7_tccr/` | TCCR 普查与端到端重组的逐图记录、门槛汇总与判决 |
| `outputs/v8_macro/` | v8 各线产物：块数据、P0 诊断、定位与解码对照、PAVE 外接、跨族迁移、作者集主表 CSV（1000 图 18000 行） |
| `reference_methods/` | [参考方法代码与来源](reference_methods/README.md) |
| `docs/` | 服务器、数据集、参考文献与实验结果 |
| `docs/temporary/` | 研究方案、实验报告和版本快照 |
| `docs/已归档的想法和文件/` | [历史研究方案](docs/已归档的想法和文件/README.md) · [已归档 Mamba 线](docs/已归档的想法和文件/mamba/README.md) · [已归档结构补偿线](docs/已归档的想法和文件/structure/README.md) |
| `tools/` | 运行记录、同步、结果整理与快照工具 |
| `tmp/` | 本地临时材料 |

在 Windows 本地拉回 v5 的小型运行记录后，使用提供 `@oai/artifact-tool` 的 Node 运行时更新 v5 工作表，并生成代码和数值结果的快照。训练图和模型权重保存在服务器。[工具说明](tools/README.md) · [v5 TXT 快照](docs/已归档的想法和文件/v5_frequency_20261109_174609.txt)

```powershell
python tools/server.py pull --version v5_frequency
node tools/update_v5_workbook.mjs
python tools/snapshot.py --version v5_frequency
```

## 参考文献

Shengyu Feng 和 Yiming Yang，*Regularized Langevin Dynamics for Combinatorial Optimization*，ICML 2025，PMLR 267，16977–16992。[正式论文页](https://proceedings.mlr.press/v267/feng25r.html) · [本地 PDF](docs/参考文献/神经组合优化/RLNN%28ICML2025%29.pdf) · [文献索引](docs/参考文献/文献索引.md) · [源码来源](docs/源码来源.md)

结构补偿一轮另整理了 2024–2026 年的 17 篇参考文献（13 篇 CCF-A）：度数偏误与高度数子群、过平滑/过挤压的残差补救、非消息传递特征通道与门控、图组合优化中的局部信息建模。PDF 按方向收在 `docs/参考文献/结构补偿/`，作者代码快照见 [参考实现](reference_methods/README.md)。[S01–S17 索引](docs/参考文献/文献索引.md#结构补偿2026-11-11)
