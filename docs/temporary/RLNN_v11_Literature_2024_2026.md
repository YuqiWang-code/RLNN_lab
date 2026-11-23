# RLNN v11 文献综述：2024–2026 CCF-A 中可迁移到 PAVE 与 1 秒搜索循环的机制

> 检索日期 2026-11-17。范围：2024-01 之后正式发表的 CCF-A（NeurIPS / ICML / ICLR / AAAI / IJCAI / KDD），主题限定为 learned local search、LNS / ruin-and-recreate、plateau & tabu、learned repair、test-time compute & budget allocation、离散采样与退火、LLM 引导的组合搜索、anytime 算法。
> 共收 **20 篇正式发表（全部 CCF-A）＋ 1 篇预印本（单列）**。PDF 全部在 `docs/参考文献/`，逐篇索引（含 DOI、页码、许可证状态）见 [`文献索引.md`](../参考文献/文献索引.md) 的「v11 文献」小节。本文件里每条结论后面的 `PDF p.X` 都指该篇本地 PDF 的页码，可直接翻页核对；摘要页写为 `p.1 摘要`。

---

## 0. 一页结论

**我们的失败不在“排序”，在“候选池”。** v10 已经测出三条硬事实：① 1 秒内的解是「1-flip + (2,1)-swap + 破坏-修复」三重邻域的不动点；② 用 KaMIS（0.1 s）输出作为 PAVE 输入，100/100 张图**零个可行 size+1 交换**；③ 学习式修复的正样本率只有 0.24%（收敛解 0.06%），远低于 1% 的训练门。这三条合起来说明：在已收敛输入上，PAVE 的候选池里根本没有“更好”的动作，因此**任何只改变“如何在既有池子里选”的机制（更强的排序器、更好的 bandit、更宽的 beam）期望增益都≈0**；只有**改变候选池本身**（禁止回到原位、扩大破坏半径、改变解所在平台）或**改变 1 秒预算的分配**的机制才有非零期望。

按这个判据，本轮 21 篇里真正值得动手的只有 5 个机制（§3 排序表），其中 3 个的成本在“一个下午 + 一次 1000 图评测”量级。**最大的单条机会是 L04（禁止最近动作的轻量 tabu）**：它不新增任何模型，只是把“修复回原解”的那些动作从池子里排除，直接对撞 v10 里“受禁破坏-修复逐位回到原解、0 胜”的失败模式。

同时有两篇论文给了我们必须写进论文的**外部锚点**：
- FLD（NeurIPS 2025）Table 1（PDF p.7）在我们**同一个基准** RB-[200–300] 上报告 FLD-EG 20.02 @ 38 s、RLSA 19.97 @ 35 s、Gurobi 19.98 @ 47.57 min、KaMIS 20.10 @ 1.40 h。也就是说：**38 秒的当前 SOTA 采样器（20.02）低于我们 1 秒的 20.0685**。这不是“我们更强”，而是“预算区间不同、数字不可比”——正好用来挡住审稿人“你为什么不和 SOTA 采样器比”的问题，也提醒我们不要拿 1 s 的数字去和 38 s 的数字互比。
- Complexity Scaling Laws（NeurIPS 2025）§5（PDF p.7）用“问题复杂度”解释子最优度随规模的变化，并明确给出与局部搜索的类比。这为我们“PAVE 的增益是输入次优度的函数”提供了理论旁证：增益随输入质量衰减不是实现缺陷，而是可预测的规模效应。

---

## 1. 检索与核验方法（可复现）

**检索链路。** dblp、AAAI OJS、ACM DL、Semantic Scholar、arXiv API 在本机全部被反爬或限流（dblp 与 AAAI OJS 返回 Anubis 校验页；arXiv API 与 S2 返回 429）。实际可用的组合是：**OpenAlex API**（无 key）按 12+15+12 组主题词检索 → 用 `primary_location` 与 DOI 前缀映射 venue → 直连 `https://arxiv.org/pdf/<id>` 或会议官方 PDF 下载 → 校验首字节 `%PDF-` 并用 PyMuPDF 打开抽全文。四轮检索（3 轮 OpenAlex 主题探测 + 1 轮针对性补漏）共 2041 条候选，venue 映射后剩 184 条 CCF-A 候选，人工筛出 21 篇。

**venue 证据分两档（写论文时必须区分）。**
- 一手证据：IJCAI 与 NeurIPS 的本地 PDF 是**带会议页脚**的正式版本。例：BALANS 首页页脚 `Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25) 2566`（PDF p.1）；UDC 为 `38th Conference on Neural Information Processing Systems (NeurIPS 2024)`（p.1）；DORA / FLD / MDNS / SolverLLM / Complexity Scaling Laws 为 `39th ... (NeurIPS 2025)`（p.1）。
- 间接证据：7 篇 AAAI 的本地 PDF 是 **arXiv 版，没有会议页眉**。这些 PDF 正文里出现的 "AAAI Conference on Artificial Intelligence" 全都在参考文献里，**不能当 venue 证据**。venue 由 DOI 前缀核实：`10.1609/aaai.v39i*` = AAAI-25，`v40i*` = AAAI-26，`v38i*` = AAAI-24。

**去重。** 与既有 102 个 PDF 和 `文献索引.md` 原有 100 行逐条比对，唯一重复项是 *Are Graph Neural Networks Optimal Approximation Algorithms?*（NeurIPS 2024，已在 `神经组合优化/Yau-2024-GNNApproximation.pdf`），未重复收录。`结构补偿/Unrealized-Expectations.pdf`（MIS 学习法 vs 经典法的诚实对照）与本轮无重叠，但它仍是我们的对照基线之一。

**代码快照。** `reference_methods/` 新增 5 个方法目录，逐个 pin 到 commit：BALANS（Apache-2.0）、DRHG（**仓库根无 LICENSE**，仅作阅读）、DORA（Apache-2.0）、FLD4CO（MIT，但仓库只有 README+LICENSE，**论文代码未公开**）、LRBS（**未见公开仓库**，按论文 §3 Algorithm 1 重写算子）。没有一处凭印象填仓库或许可证。

---

## 2. 逐篇：机制 / 与 PAVE 或搜索循环的关系 / 改我们代码哪一处 / 预期收益量级 / 成本 / 风险

> 收益量级一律用我们自己的刻度标注：±0.01 属于 1000 图配对比较的噪声带；+0.05 是我们已测到的“捕获摊薄”量级；+0.222 是 PAVE 在 0.5 s 输入上的实测增益；+0.710 是贪心输入上的实测增益。所有“预期”都是估计，不是测量值。

### L 组：学习型局部搜索

#### L01 BALANS（IJCAI 2025，CCF-A）— 用 bandit 在线选 (destroy, repair) 算子对

**机制。** 把 ALNS 的每个 (destroy, repair) 算子**对**当作一个 bandit arm，destroy 尺寸不做超参搜索，而是让同一个算子在 portfolio 里以不同尺寸重复出现、各自当 arm（PDF p.2–p.3, §3）。MAB 在线选择，**不需要任何离线训练或监督标签**；reward 被离散成 4 级，直接对齐 acceptance 结果（新解=历史最优 / 比上一步好 / 非最好但被接受 / 被拒）。acceptance 用 HC 或 SA，作者强调“把状态探索与邻域探索分开处理”是本文的关键新颖点。base solver 可换 SCIP / Gurobi（solver-agnostic，Q3 消融）。

**与我们的关系。** 这是 PAVE 学习排序器（PAVEV3）的**零训练对照**。我们现在的排序器要离线学、正样本率只有 0.24%/0.06%；BALANS 说明“同样的算子池 + 在线 bandit + 离散 reward”也能拿到增益，且不需要正样本。如果我们做一次 bandit 臂，能把“排序器的价值来自特征还是来自学习”这件事切开。

**改哪一处。** `models/mis_v3_search.py:340 platform_search()`：现在是 `choose="coverage"` 或 ranker 单点决策，可在 `outer_rounds=8` 的循环里插入一个 arm 选择器（arm = (destroy 半径, repair 策略)），reward 用 `repair()` 前后的 `bit_count()` 差分。动作池仍由 `models/mis_v2_exchange.py:86 actions()` 提供。

**预期收益。** 在已收敛输入上 **≈0**（池里没动作，选什么都一样）；在弱输入（贪心起点）上 +0.2~+0.6 量级，与我们已知的 PAVE 增益同阶。真正价值是**诊断**：如果 bandit 打平排序器，说明排序器没有存在必要。

**成本。** 极低：不需要训练，不需要 GPU，只需在 `platform_search` 里加 ~50 行 UCB 记账。

**风险。** ① reward 太稀疏（我们 8 轮里多数轮次 gain=0）会让 UCB 退化成轮转；BALANS 的 4 级 reward 正好是为这种稀疏情形设计的，可借用。② 与我们 v10 已测的“学习式退火全负”不冲突，但也不构成新证据。

#### L02 ReinforceNS（IJCAI 2024，CCF-A）— RL 选邻域 + 邻域预缩减

**机制。** TTP（TSP+背包）上，先用图/邻域预缩减砍掉不可能改进的邻域，再用 RL 在**多起点邻域搜索**里选邻域；后优化分 item flipping 与 route refining 两阶段（PDF p.1 摘要）。60 个实例（76–33810 城）胜过 3 个 SOTA，18 个公开实例里刷新 12 个。

**与我们的关系。** 与我们同构的地方是“**选邻域而不是选解**”。但它的增益来源是“问题规模大、邻域巨大，选对邻域能省预算”；我们的规模（200–300 点、度 ~15）邻域本来就小，v10 已经证明候选池在收敛后为空。

**改哪一处。** 若要跟，只跟“预缩减”那一半：`models/mis_v2_exchange.py:48 _exchanges()` 里对 `removed` 的组合枚举加度数/覆盖剪枝。RL 那一半不建议（理由同上）。

**预期收益。** 剪枝只省时间、不改解质量，期望 **+0.00**，收益体现在同样 1 s 内多跑几轮。

**成本。** 低（纯 CPU 枚举剪枝）。**风险。** 剪枝会改变候选池顺序，可能让 `cap=32` 截断掉本来能用的动作——这正是 `candidate_cap` 的已知敏感点，必须配对 ablation。

#### L03 NS4S（IJCAI 2025，CCF-A）— LLM 生成“算子权重更新规则”

**机制。** JSP/FJSP 上，LLM 不是直接给解，而是**评估邻域并动态调 operation 权重**，再配 VeEvo（verification evolution）抑制幻觉、演化出高质量的权重更新规则（PDF p.1 摘要）。349 个实例；JSP Taillard 平均 gap 10.46%→1.35%，FJSP Brandimarte 13.24%→0.05%。

**与我们的关系。** 这是“算子工厂”派的第二条路线（第一条是 L09）。对我们的价值不在收益，在**写法**：它示范了如何把“LLM 产出的规则”做成可验证、可回滚的组件，而不是把 LLM 塞进内循环。

**改哪一处。** 不进内循环。若要试，只在外层调参脚本里生成 `platform_search` 的超参策略（`outer_rounds` / `candidate_cap` 随图密度变化）。

**预期收益。** 内循环 **0**；外层调参可能 +0.01~+0.02。**成本。** 高（需要 LLM 调用与验证回路）。**风险。** 我们 1 s 的预算里没有“调用 LLM”的位置；任何进内循环的尝试都会把 1 s 变成 10 s。

#### L04 AlphaZero + Tabu（IJCAI 2024，CCF-A）— 禁止“最近做过的动作”的轻量 tabu ★

**机制。** 在极值图（无 C3/C4 的最大边数图）搜索上，作者用的 tabu 不是经典的“禁止访问过的状态”，而是**禁止最近做过的动作**：把最近翻过的边放进 FIFO 队列，队列长度 h 是唯一超参（PDF p.5, §5 + Algorithm 1）。论文的结论很直白：AlphaZero 与 tabu **各有胜负**，而 tabu 简单且强；Incremental Tabu / Incremental AlphaZero 用更小 n 找到的好图解启更大 n（课程式热启动）。

**与我们的关系。** 这条直接对撞 v10 的失败模式：**“受禁破坏-修复 0 胜且逐位回到原解”**。我们当时的“受禁”是禁止**状态**，所以修复算子把解又拉回不动点；L04 的 tabu 禁止的是**动作**，同一批动作不会被重复选中，平台搜索就能在等规模解之间走动。这与 PAVE 的 `extend` 阶段假设（“在平台上重复 size+1 交换直到不动点”）是同一个问题的两种解法。

**改哪一处。** 最小改动点有两处，二选一：
- `models/mis_v3_search.py:277 rollout()`：给 `candidate_actions(graph, work, cap=candidate_cap, tabu=(action,))` 里的 `tabu` 从“单个动作”扩成“最近 h 个动作的集合”（FIFO）。
- `models/mis_v3_search.py:340 platform_search()`：在 `outer_rounds=8` 的循环里维护动作 FIFO，喂给 `repair()` / `candidate_actions()`。

**预期收益。** 这是我们估计**唯一可能突破 +0.01 的三条之一**：目标是把“平台搜索在等规模解之间走动的轮数”从 0 提到 >0。若走动成立，增益落在 **+0.02~+0.08**（介于捕获摊薄 +0.050 与 PAVE 在 0.5 s 输入上的 +0.222 之间）。**若走动不成立（接受率仍为 0），收益严格为 0，不会变负**——这是它风险低的原因。

**成本。** 极低：~40 行代码，无训练，无新模型，1000 图评测一次即可判定；1 s 预算下 FIFO 查找可忽略。

**风险。** ① 我们的动作空间是“交换”而非“翻边”，FIFO 的键要选对（建议用 `Exchange.removed` 的位掩码，而不是整个 action）；② h 是唯一超参，若 h 太小会退化成无 tabu，太大则锁死池子——论文用的 h 需要按图规模缩放，建议 h ∈ {8, 16, 32} 三档扫。

#### L05 LRBS / Scaling CO Neural Improvement Heuristics（AAAI 2025，CCF-A）— 推理期 beam + 限步 rollout ★

**机制。** 对 improvement 策略做**推理期 beam search**：beam 宽度 β（每层保留的活跃节点数）、每个节点扩展 α 个子节点，子节点不用完整 rollout 打分（那等于 MCTS 的 simulated playout），而是用 **ns 步限步 rollout** 估值，避免反传开销；再用同一搜索过程做 offline/online adaptation（PDF p.3, §3 + Algorithm 1）。

**与我们的关系。** 这正是 PAVE `extend` 的“宽度化”。PAVE 的部署参数是 `horizon=2`（`models/eval_pave_external.py:46 PAVE_ARGS`；任务书里记的 horizon=1 是更早的一档），本质仍是单点贪心；LRBS 把它变成 β×α 的宽度搜索，并用便宜 rollout 排序。**但要先接受第 0 节的判据**：如果候选池里没有可行交换，加宽 beam 只是把“没有动作”这件事乘上 β×α。

**改哪一处。** `models/mis_v3_search.py:277 rollout()`（`horizon` 从 1/2 提到 β）与 `:340 platform_search()` 的 macro 选择；动作池仍用 `models/mis_v2_exchange.py:86 actions(cap=32)`，把 `improving_cap=48` 提到 128 以容纳 beam 子节点。

**预期收益。** 已在 `reference_methods/LRBS/operators/lrbs_beam.py` 的玩具实验里验证方向：200 点随机图、贪心起点，β=1/α=1/ns=0（等价 PAVE horizon=1）只能 +0~+1，β=4/α=5/ns=4/rounds=2 能 +2。在 **RB 已收敛解上期望 ≈0**（v10：KaMIS 输入 100/100 图零交换），在**弱输入上**可能放大现有 +0.222。

**成本。** 低（纯 CPU 枚举 + 多点 rollout 估值，可在 1 s 内做）。**风险。** 这是本轮**最容易被误用**的机制：它在弱输入上有效，很容易被误读成“在收敛解上也有效”。kill criteria 必须写死（见 §3）。

#### L06 DRHG / Destroy and Repair Using Hyper-Graphs（AAAI 2025，CCF-A）— 破坏后压成超边 ★

**机制。** destroy → 图缩减 → hyper-graph 求解 → 图恢复。关键是把**未被破坏的连续段收缩成一条超边**（只用两个端点表示，"remove the middle nodes"），于是 repair 模型的输入规模只取决于破坏程度、与问题规模无关；用监督学习重连孤立点与固定超边（PDF p.2–p.3, §3.1–3.2）。TSP 100–10K SOTA。

**与我们的关系。** 我们的 `mis_lns.py:74 repair(selected, region, neighbours)` 是“破坏一个 region 再贪心重构”，破坏半径固定且小；DRHG 给的是**破坏半径与 repair 成本解耦**的做法。对我们最有用的不是 hyper-graph 模型，而是这句：**repair 的代价应该只依赖破坏程度**。这让我们能在 1 s 里做“半径更大但成本不涨”的破坏。

**改哪一处。** `models/mis_lns.py:74 repair()`：破坏集合从“1 跳邻域”改为“k 跳 + 段收缩表示”；`models/mis_v3_search.py:247 repair()` 的 `anchor` 参数可以承载超边端点。

**预期收益。** 只有破坏半径真的变大才有增益：估计 **+0.02~+0.10**，但**方差大**（依赖 RB[200–300] 上 k 跳破坏是否还能保持 repair 的可解性；v10 已测“大移动/图缩减”全负，k 跳破坏与它不是同一件事，但同属“更大动作”家族，必须当作**高风险**）。

**成本。** 中：破坏枚举 + 段收缩表示要重写，约 150–250 行；若引入学到的 repair 模型则是高成本（且 v10 已证正样本率不够）。

**风险。** 这是本轮**唯一与 v10 已作废方向（大移动）相邻**的机制，必须先用“k 跳破坏后可行 repair 的比例”做一次纯统计的门诊（不需要训练），比例 <5% 就直接放弃。

#### L07 LNS2+RL（AAAI 2025，CCF-A）— “先学后快”的自适应切回

**机制。** MAPF 上，早期迭代用 MARL 重规划（消解冲突远多于 PP），后期按“冲突数（CP 计数）+ 时间阶段”**自适应切回**快速启发式 PP+SIPPS，用质量换效率（PDF p.3, §3 + Algorithm 1）。

**与我们的关系。** 与我们 PAVE“增益是输入次优度的函数”**同构**：学到的算子只在早期/次优阶段值钱，收敛后应当切回便宜算子。这条给我们论文提供了一个**外部同类证据**：不是只有我们观察到“学到的东西在收敛后没用”。

**改哪一处。** `models/mis_v8_capture.py:107 run(deadline, decode, ...)`：在捕获循环里按剩余预算比例切换“学到的策略 / 便宜规则”。

**预期收益。** 主要是**省预算**（把 1 s 里被学到算子吃掉的时间还给搜索），间接 +0.01~+0.05，对齐我们的捕获摊薄 +0.050。**成本。** 低。**风险。** 切回点需要调，调不好会两边都丢；建议直接复用它的判据（冲突/改进计数阈值），不要自创。

#### L08 COMBHelper（AAAI 2024，CCF-A）— GNN 打分剪掉不可能进解的节点

**机制。** GNN 先给节点打分、剪掉不太可能进解的节点，把缩减后的搜索空间喂给传统 CO 算法；另有 KD 模块与问题特定 boosting 模块；传统算法至少快 2 倍（PDF p.1–p.2, §1）。

**与我们的关系。** 与 v10 已测的“大移动/图缩减”直接同类：RB[200–300] 上**度 ≤3 的顶点占比 0%**，能剪掉的只有极少数；kernel 还剩 88.1%。也就是说这条在**我们的图分布上结构性地无效**。

**改哪一处。** 不建议改。若要记录，只记在 Related Work 里说明“为什么剪枝不适用于 RB[200–300]”。

**预期收益。** **≈0 或负**（剪掉的可能正是需要交换的点）。**成本。** 低（但会把预算从搜索挪到 GNN 前向）。**风险。** 已被 v10 独立证伪两次，不必再试。

#### L09 Hercules（KDD 2025，CCF-A）— LLM 自动生成启发式 + 可靠性判别

**机制。** LLM 自动生成启发式（Heuristic Generation），CAP + PPP 两个机制提升预测准确性、识别不可靠预测，PPP 版 Hercules-P 更省算力；4 个 HG 任务、5 个 COP、8 个 LLM 上超 SOTA（PDF p.1 摘要）。代码在 Zenodo（10.5281/zenodo.15462797）。

**与我们的关系。** “算子工厂”的当前上限，属于**第二创新点**的长线候选，不是 v11 的即时选项。

**改哪一处。** 不进内循环；可在外层生成“破坏-修复算子族”，再用 1 s 评测筛。

**预期收益。** 长线；短期 0。**成本。** 高。**风险。** 生成出的算子极可能在 RB 上等价于我们已有的三重邻域（v10 已证该邻域是不动点）。

### B 组：预算与自适应计算

#### B01 DORA / Every Rollout Counts（NeurIPS 2025，CCF-A）— 方向级最优预算分配 ★

**机制。** 把 test-time search 建模成资源分配，最大化“至少一次成功”的概率。Proposition 2：**solution-level 分配次优**——按候选数 k_j 与 PRM 分数 e^{R_j} 加权的 Eq.8 在 k_j 不等时不是最优；最优是 **direction-level 的 softmax 分配**：`B_j = N·e^{R_j}/Σ_l e^{R_l}`（Eq.9），κ→0 为默认设定（PDF p.4–p.6, §3.1）。代码 WangXinglin/DORA（Apache-2.0）。

**与我们的关系。** 我们的 1 s 是**固定切分**（捕获 808 步 + decode + PAVE 预留），v10 已经证明这个切分是“捕获摊薄 +0.050”的来源，也是时钟 bug 的来源（`run()` 的 deadline 需要每块 `torch.cuda.synchronize()` 才生效，K=200 真实单步 2.96 ms）。DORA 给的是一条**按方向而非按候选数**分配预算的原则：如果我们的候选方向（不同种子/不同 recombine 组合）质量不等，就应当按 e^{R} 加权而不是平分。

**改哪一处。** `models/mis_v8_capture.py:107 run(self, deadline, decode, ...)` 的预算分段，以及 `models/eval_v8_recombine.py:74 run_arm(...)` 里 `reserve` 的分配；排序分数直接复用 ranker 输出。

**预期收益。** **+0.01~+0.05**，与捕获摊薄的量级一致；这是“不改动作池也能拿分”的三条之一。**成本。** 低（改分段权重，无需训练）。**风险。** 我们的“方向”质量估计（R_j）噪声大，softmax 会放大噪声；建议先做 κ 扫描，κ 大时退化为平均分配，作为安全兜底。

#### B02 Threshold UCT（AAAI 2025，CCF-A）— 成本约束 MCTS + Pareto 曲线

**机制。** 在 CMDP（成本约束 MDP）上做 MCTS：节点上维护 cost-utility **Pareto 曲线**并做完整 Bellman 更新，配 threshold-based action selection 与 threshold 更新规则，避免过度保守；超参少（PDF p.1–p.4）。

**与我们的关系。** 把“1 s 墙钟”形式化为硬成本约束的 anytime 决策——这正是我们需要的语言：**不是“跑到时间就停”，而是“在成本约束下最大化期望增益”**。但它的实验域是自主车导航，与 MIS 差得远，直接搬算法不现实。

**改哪一处。** 概念层：`platform_search()` 的 `remaining = (outer_rounds - step)/outer_rounds` 现在是一个手写的线性衰减，可替换为“Pareto 前沿上的 threshold 判据”。

**预期收益。** +0.00~+0.02（主要是**可解释性**与论文的 anytime 叙事）。**成本。** 低。**风险。** 域差太大，容易写成“借了个名词”，审稿人会问为什么不直接用 budgeted bandit。

#### B03 Complexity Scaling Laws（NeurIPS 2025，CCF-A）— 子最优度随规模的可预测增长

**机制。** 以问题复杂度（解空间大小、表示空间大小）建 scaling law；TSP 上子最优度随节点数/维度**可预测**增长，且与用 RL 还是 SFT 无关；§5 明确给出与**局部搜索**的类比（对代价景观做更简单的梯度下降产生同样趋势），高维下问题复杂度趋近平稳（PDF p.7, §5；Theorem 5/6）。

**与我们的关系。** 为我们论文的核心命题“PAVE 的增益是输入次优度的函数”提供**理论旁证**，也给“为什么 1 s 之后增益消失”一个可写进 Discussion 的解释。它**不提供算法**。

**改哪一处。** 不改代码。写论文时用来支撑 §“gain as a function of input suboptimality”。

**预期收益。** 0（对代码），但对**论文的审稿防御**价值高。**成本。** 0。**风险。** 不能把它的结论外推到 MIS——论文的实验是 TSP，我们的图分布（度 ~15、度≤3 占 0%）与 TSP 的代价景观不同，措辞必须限定。

#### B04 SolverLLM（NeurIPS 2025，CCF-A）— 免训练的 MCTS 变体

**机制。** 免训练：把问题生成数学形式化 → 转成 solver-ready 代码，MCTS 引导；对经典 MCTS 加三处改动：dynamic expansion、prompt backpropagation、uncertainty backpropagation（PDF p.1 摘要）。

**与我们的关系。** 是我们“测试期算力调度”的对照样本，但依赖 LLM 与代码执行，与 1 s 内循环不相容。

**改哪一处。** 不改。**预期收益。** 0。**成本。** 高。**风险。** 会被审稿人拿来问“为什么不用 LLM”，需要在 Related Work 里明确“预算量级不同”。

#### B05 Parallel CPU-GPU Batching（AAAI 2026，CCF-A）— 搜索在 CPU、评估在 GPU

**机制。** cost-bounded depth-first search（CB-DFS）框架：树搜索在 CPU 并行、启发式评估在 GPU 并行批处理（IDA*、Budgeted Tree Search），解决“DFS 必须等启发式算完才能推进”的阻塞；3×3 魔方与 4×4 STP 验证（PDF p.1 摘要）。

**与我们的关系。** 我们的 1 s 瓶颈恰好是 **CPU 侧的交换枚举/修复 与 GPU 侧的策略前向交替**（这也是时钟 bug 的来源）。这条给了“异步流水线”的工程模板，但它是 DFS 框架，我们是迭代改进。

**改哪一处。** 若做，改 `models/mis_v8_capture.py:107 run()` 的块调度（让 CPU 侧的 repair 与 GPU 侧的前向重叠），属于**系统优化**，与 v10 的 systems 叙事一致。

**预期收益。** 纯系统：同 1 s 内多跑 5–20% 步数 → 间接 **+0.01~+0.03**。**成本。** 中高（改调度，容易再次踩时钟 bug）。**风险。** v10 的时钟 bug 就出在“以为在重叠、其实被同步阻塞”，再来一次异步改造很容易复发。

### D 组：离散采样与退火

#### D01 FLD（NeurIPS 2025，CCF-A）— 截断 α-stable 噪声与多项式时间逃逸

**机制。** 用对称 α-stable（SαS）噪声替换高斯噪声（带截断），α=2 退化为 Langevin dynamics；理论主张：高斯噪声下逃逸时间随势垒**高度指数**增长，FLD 只随势垒**宽度多项式**增长；两种形态 FLD-EG（显式梯度采样）/ FLD-IG（隐式梯度 + RL）（PDF p.2, §1；p.7, §4.3）。超参表（p.27）：MIS RB-[200–300] 用 τ0=0.01, d=5, K=200, T=300, λ=1.02, d_noise=9, **α=1.2**。

**与我们同基准的硬数字（Table 1, PDF p.7）**：Gurobi 19.98 @ 47.57 min；KaMIS 20.10 @ 1.40 h；**FLD-EG 20.02 @ 38 s**；RLSA 19.97 @ 35 s；SDDS 19.62 @ 20 s；RLNN 19.52 @ 1.64 min；FLD-IG 19.72 @ 1.08 min；DIFUSCO 18.52 @ 16.05 min；INTEL 18.47；DGL 17.36。**38 秒的 SOTA（20.02）低于我们 1 秒的 20.0685**——这说明预算区间不同、数字不可比，而不是我们更强。

**与我们的关系。** 我们的解是“三重邻域不动点”，逃逸需要“能跨过平台”的噪声。FLD 的贡献是**噪声族的形状**（重尾）而不是“更好的 sampler”。我们 1 s 内的策略是离散翻转、没有连续松弛，所以直接搬 FLD 的 Langevin 更新不成立；能借的是**在动作选择上加一个重尾随机扰动**（等价于“偶尔选一个 coverage 很差但能打开局面的交换”）。

**改哪一处。** `models/mis_v3_search.py:247 repair()` 与 `:340 platform_search()` 的 `choose` 通道：把“规则/ranker 选动作”改成“ranker 分数 + 重尾噪声（α=1.2，按 d_noise 截断）”。算子已重写在 `reference_methods/FLD4CO/operators/fld_sas.py`（129 行，自检：α=2 方差 2.001；α=1.2 的 99.9 分位 114.8 vs 高斯 4.66；同步长下双井势中位首次跨井 24 步 vs 高斯 53 步）。

**预期收益。** **不确定，可能在 0 与负之间**：v10 已测“学习式退火温度单调变差”，重尾噪声是另一条自由度但在收敛解上同样可能只是把好解打散。乐观情形 +0.02~+0.05。

**成本。** 低（算子已写好）。**风险。** ① 1 s 预算下重尾噪声会浪费步数；② 需要重新校准 `candidate_cap`（噪声选中的动作可能被 cap 截断）；③ 很容易变成“又一次退火实验”，而 v10 已经否掉过一次。

#### D02 Cyclical Discrete Sampling（NeurIPS 2024，CCF-A）— 循环步长 + 循环 balancing 的自动调参

**机制。** 三件套：循环步长表（大步发现新模式、小步利用）、循环 balancing 表（保证给定步长下 proposal 是 "balanced" 的、链高效）、自动调参；给出一般离散分布的非渐近收敛与推断保证；针对多峰离散分布被困局部模式（PDF p.1, §1 + 摘要）。

**与我们的关系。** 我们不需要“从分布里采样”，需要“从平台里出来”。它的循环步长表是一个**免调参的退火表**，可以直接借给 D01 的噪声幅度调度。**改哪一处。** `platform_search()` 的 `remaining` 衰减 → 循环表。**预期收益。** 与 D01 合并计 +0.02~+0.05。**成本。** 极低。**风险。** 我们的“步长”是离散动作数而非温度，循环表要重新定义。

#### D03 MDNS（NeurIPS 2025，CCF-A）— 带权采样的随机最优控制

**机制。** 在离散状态空间上学神经采样器（π ∝ e^{-U}，U 已知至归一化常数），用连续时间 Markov 链的随机最优控制对齐两条路径测度，给出一个学习目标族；高维多峰下大幅超学习型基线（PDF p.1 摘要）。代码 yuchen-zhu-zyc/MDNS（**未声明许可证**）。

**与我们的关系。** 说明“采样器”这条路要付的训练成本，且它的目标是采样而非优化。**改哪一处。** 不改。**预期收益。** 0（1 s 内不可能训练）。**成本。** 高。**风险。** 会被误当成“更好的 sampler = 更好的 solver”。

### N 组：神经组合优化

#### N01 UDC（NeurIPS 2024，CCF-A）— Divide-Conquer-Reunion 训练

**机制。** GNN 做全局划分 + 定长子路径求解器，Divide-Conquer-Reunion 训练消除“子问题划分策略次优”的负面影响；10 个大规模 CO 问题（PDF p.1 摘要）。

**与我们的关系。** 我们的图只有 200–300 点，不需要分治；v10 的“大移动/图缩减”线也说明分治在 RB 上无收益。**改哪一处。** 不改。**预期收益。** 0。**成本。** 高。**风险。** 无（不跟进）。

#### N02 DOGE-Train（AAAI 2024，CCF-A）— 学习型更新步跳出基础求解器的次优不动点

**机制。** GNN 预测 Lagrange 分解算法的参数并端到端可微，保持对偶可行与下界不降；用**非参数 GNN 更新步**跳出基础求解器的次优不动点；无监督损失；~10k 参数；any-time 性能优于商用求解器（PDF p.1 摘要）。

**与我们的关系。** “跳出基础求解器的次优不动点”与我们要解决的问题**字面上完全一致**，但它的载体是 Lagrange 对偶（连续、可微），我们的载体是离散独立集。它的可迁移部分是**无监督损失 + 极小参数**的工程哲学（~10k 参数 vs 我们的排序器），适合作为“如果要做小模型修复器”的模板。

**改哪一处。** 不直接改；若做第二创新点，参考它的无监督目标构造。**预期收益。** 长线 0；**成本。** 高。**风险。** 对偶可行与离散可行是两件事，不能类比太远。

#### N03 LLMs as End-to-end CO Solvers（NeurIPS 2025，CCF-A）— 自然语言直接映射解

**机制。** 两阶段：SFT 从领域求解器学解的模式，FOARL（feasibility-and-optimality-aware RL）压约束违反；7 个 NP-hard 问题，7B 模型可行率高、平均 gap 1.03–8.20%，超 GPT-4o/DeepSeek-R1（PDF p.1 摘要）。

**与我们的关系。** gap 1–8% 与我们的 20.06（相对 KaMIS 20.10 的 0.2%）不在同一量级，属于 Related Work 的对照。**改哪一处。** 不改。**预期收益。** 0。**成本。** 极高。**风险。** 无（不跟进）。

### PRE01 预印本（单列）：RL 在局部搜索邻域选择里到底有没有用

**机制。** 系统评估 local search 元启发式里的 RL 型邻域选择策略——multi-armed bandits（UCB、ε-greedy）等，并明确指出“RL 在局部搜索里的有效性其实研究不足”（PDF p.1 摘要）。作者 ULB / UCLouvain / KU Leuven / CETIC，2026-01 预印本，13 页，未见正式发表。

**与我们的关系。** 与本项目“学习排序器正样本率 0.06–0.24%”的诊断**直接对话**：如果连 bandit 都打不过简单启发式，我们的排序器大概率也一样。它给了一次**现成的实验协议**（同一套 local search，只换邻域选择器）。

**改哪一处。** 不改。做“bandit vs ranker vs coverage 规则”三臂消融时，协议直接引用它。**预期收益。** 0（对代码）；对**消融的说服力**价值高。**成本。** 0。**风险。** 预印本未经同行评审，只能当协议参考，不能当证据。

---

## 3. 最值得试的 5 个机制（排序）

排序依据：**能否改变候选池或改变预算分配**（第 0 节判据）> 实现成本 > 与 v10 已有证据的冲突程度。所有条目都限定在 1 s 预算、RB[200–300]、作者 1000 图上评测；评测脚本沿用 v10 的配对 bootstrap（`models/eval_pave_external.py:96 paired_bootstrap`）。

| 排名 | 机制 | 一句话做什么 | 改哪一处 | 预期收益 | 成本 | 一行 kill criteria |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **L04 动作级 tabu**（IJCAI 2024） | 把 FIFO 队列里最近做过的那批交换从候选池里排除，让平台搜索在等规模解之间真的走动 | `models/mis_v3_search.py:277 rollout()` 的 `tabu=` 参数扩成集合；`platform_search()` 维护 FIFO | +0.02~+0.08（若走动成立）；不走动则严格 0 | 极低（~40 行，无训练） | 1000 图上 `outer_rounds=8` 内“接受率（等规模新平台解）”仍为 0 → 立刻停，不调 h |
| 2 | **B01 方向级预算分配**（NeurIPS 2025） | 把 1 s 里的捕获/搜索/修复切分从固定比例改成按 e^{R_j} 的 softmax 分配 | `models/mis_v8_capture.py:107 run(deadline, decode, ...)` 的分段；`models/eval_v8_recombine.py:74 run_arm()` 的 `reserve` | +0.01~+0.05（对齐捕获摊薄量级） | 低（改权重，无训练） | κ 扫描中最好点相对固定切分 < +0.01（配对 CI 跨 0）→ 停 |
| 3 | **L06 破坏半径与 repair 成本解耦**（AAAI 2025） | 破坏集合从 1 跳改 k 跳，用段收缩把 repair 成本钉在破坏程度上 | `models/mis_lns.py:74 repair()`；`models/mis_v3_search.py:247 repair(anchor=)` | +0.02~+0.10，方差大 | 中（150–250 行） | **先做门诊**：k=2 跳破坏后“存在可行 repair”的比例 <5% → 不写代码就停 |
| 4 | **L01 bandit 选算子对**（IJCAI 2025） | 零训练的 UCB/ε-greedy 替代学习排序器，用 4 级离散 reward | `models/mis_v3_search.py:340 platform_search()` 的 `choose` 分支 | 收敛解 ≈0；弱输入 +0.2~+0.6（诊断价值 > 收益） | 低（~50 行） | 三臂（bandit / ranker / coverage）在 1000 图上配对比较，bandit 与 ranker 差异 CI 跨 0 → 结论写进论文，不加权 |
| 5 | **L05 限步 rollout 的 beam 宽度化**（AAAI 2025） | 把 PAVE 的 horizon=1 提到 β×α，用 ns 步 rollout 排序子节点 | `models/mis_v3_search.py:277 rollout()` 的 `horizon`；`improving_cap` 48→128 | 弱输入可放大 +0.222；**收敛解 ≈0** | 低（复用 `reference_methods/LRBS/operators/lrbs_beam.py`） | 若第 1 项（tabu）失败（接受率 0）→ 这一项同时作废，因为池子空时 beam 无用 |

**第二梯队（值得记录，不建议现在动手）**：D01 FLD 重尾噪声（算子已就绪，但 v10 已否掉一次退火家族，且容易把好解打散）、B02 Threshold UCT（概念有用，域差太大）、B05 CPU-GPU 异步（会再次踩时钟 bug 的坑）、L07 自适应切回（与 B01 重叠，可合并做）。

**明确不跟进**：L03 / L09（LLM 进内循环与 1 s 不相容）、L08（剪枝在 RB 上结构性无效，已被证伪两次）、N01 / N03 / D03 / B04（成本与预算量级不匹配）、L02 的 RL 部分（理由同 L01）、B03 的算法部分（只取叙事）。

---

## 4. 与 v10 已作废方向的重叠检查

写这一节是为了让下一轮不再重复踩坑。v10 已判负的十三条线里，与本轮文献**正面重叠**的有三组：

1. **“学习式修复/更宽搜索” ↔ L05 / L08 / N02**：v10 的“学习式修复正样本率 0.24%（收敛解 0.06%）”与“KaMIS 输入 100/100 图零交换”是同一件事的两面。任何以“更好的排序/更宽的 beam”为核心卖点的机制，都必须先证明**候选池非空**，否则收益恒为 0。
2. **“退火/噪声” ↔ D01 / D02**：v10 的“学习式退火温度单调变差”否掉的是**温度调度**，不是**噪声族形状**。FLD 的重尾噪声严格来说是新自由度，但预期收益低且实验成本不低，列第二梯队。
3. **“大移动/图缩减” ↔ L08 / N01 / L06**：前两个已被证伪；L06 是**唯一**与“大移动”相邻但机制不同的（它放大**破坏半径**而不是**动作尺寸**），因此保留但排在第三、且必须先做门诊。

**没有重叠、真正新的东西**：L04（动作级 tabu）、B01（方向级预算分配）、L07（自适应切回）、B02（成本约束的 anytime 形式化）。这四条的共同点是**不新增模型**，只改“选什么/在什么时候选/给多少预算”。

---

## 5. 核验记录（每条结论的出处）

| 文档 | 本文件中引用的证据 | 位置 |
| --- | --- | --- |
| BALANS | ALNS(MIP) 元求解器、arm=(destroy,repair)、4 级 reward、HC/SA acceptance、solver-agnostic 消融 | PDF p.2–p.3, §3；venue 页脚 p.1（IJCAI-25, 2566） |
| ReinforceNS | 邻域预缩减 + RL 选邻域 + 两阶段后优化；60 实例/12 个刷新上界 | PDF p.1 摘要；页脚 p.1（IJCAI-24, 7038） |
| NS4S | LLM 做邻域评估、VeEvo 抑制幻觉；JSP 10.46%→1.35% | PDF p.1 摘要；页脚 p.1（IJCAI-25, 8687） |
| AlphaZero+Tabu | 禁止最近动作的 FIFO tabu、唯一超参 h、Incremental 热启动、“tabu 简单且强” | PDF p.5, §5 + Algorithm 1；页脚 p.1（IJCAI-24, 6985） |
| LRBS | 推理期 beam β、子节点 α、ns 步限步 rollout、online adaptation | PDF p.3, §3 + Algorithm 1；venue 由 DOI 10.1609/aaai.v39i25.34921 |
| DRHG | destroy→缩减→超边求解→恢复；超边只保留两端点；模型输入与问题规模无关 | PDF p.2–p.3, §3.1–3.2；venue 由 DOI 10.1609/aaai.v39i17.34018 |
| LNS2+RL | 早期 MARL、后期按冲突数/时间阶段切回 PP+SIPPS | PDF p.3, §3 + Algorithm 1；venue 由 DOI 10.1609/aaai.v39i22.34501 |
| COMBHelper | GNN 打分剪枝 + KD + boosting，传统算法至少快 2 倍 | PDF p.1–p.2, §1；venue 由 DOI 10.1609/aaai.v38i18.30070 |
| Hercules | CAP + PPP、zenodo 代码、4 HG 任务/5 COP/8 LLM | PDF p.1 摘要；KDD '25 版权块含 DOI 10.1145/3711896.3736923 |
| DORA | Eq.8 解级分配次优、Eq.9 方向级 softmax、κ→0 | PDF p.4–p.6, §3.1（Proposition 2）；页脚 p.1（NeurIPS 2025） |
| ThresholdUCT | CMDP + Pareto 曲线 + threshold action selection | PDF p.1–p.4；venue 由 DOI 10.1609/aaai.v39i25.34858 |
| ComplexityScaling | 复杂度 scaling law、§5 与局部搜索的类比、Theorem 5/6 | PDF p.7, §5；页脚 p.1（NeurIPS 2025） |
| SolverLLM | 免训练、dynamic expansion / prompt & uncertainty backprop | PDF p.1 摘要；页脚 p.1（NeurIPS 2025） |
| ParallelCpuGpu | CB-DFS、CPU 并行树搜索 + GPU 批处理评估、3×3 魔方/4×4 STP | PDF p.1 摘要；venue 由 DOI 10.1609/aaai.v40i43.41019（= AAAI-26） |
| FLD | Table 1 全表数字、α=1.2 / d_noise=9 / K=200 / T=300、逃逸时间理论 | PDF p.7, Table 1；p.2, §1；p.27, Table 6；页脚 p.1（NeurIPS 2025） |
| Cyclical | 循环步长表/循环 balancing 表/自动调参与非渐近保证 | PDF p.1, §1 + 摘要；页脚 p.1（NeurIPS 2024） |
| MDNS | 随机最优控制目标族、离散状态空间神经采样器 | PDF p.1 摘要；页脚 p.1（NeurIPS 2025） |
| UDC | Divide-Conquer-Reunion 训练 | PDF p.1 摘要；页脚 p.1（NeurIPS 2024） |
| DOGE-Train | 端到端可微 Lagrange 参数、~10k 参数、无监督损失、跳出不动点 | PDF p.1 摘要；venue 由 DOI 10.1609/aaai.v38i18.30048 |
| LLMEnd2EndCO | SFT + FOARL、7 个 NP-hard、gap 1.03–8.20% | PDF p.1 摘要；页脚 p.1（NeurIPS 2025） |
| PRE01 | bandit（UCB、ε-greedy）邻域选择评估、“研究不足” | PDF p.1 摘要；无会议页脚（预印本，未核实到正式发表） |

**仓库与许可证核验**（均通过 `git ls-remote` / GitHub API / 实际 clone 确认，未凭印象填写）：`skadio/balans`（Apache-2.0，commit 9dbf48f）、`CIAM-Group/DRHG`（**无 LICENSE 文件**，commit 1e28f20）、`WangXinglin/DORA`（Apache-2.0，commit 150a0f7）、`Thinklab-SJTU/FLD4CO`（MIT，commit 083101d，**仅 README + LICENSE**）、`marmotlab/LNS2-RL`（MIT，仅记录在索引，未做快照）、`yuchen-zhu-zyc/MDNS`（**未声明许可证**）。LRBS 的 arXiv 摘要只给 anonymous URL，因此 `reference_methods/LRBS/` 里没有 `source/`，只有按论文 §3 重写的算子。

---

## 6. 文件清单

- PDF（21 篇）：`docs/参考文献/学习型局部搜索/`（9）、`预算与自适应计算/`（5）、`采样与退火/`（3）、`神经组合优化/`（3）、`预印本/`（1）
- 索引：`docs/参考文献/文献索引.md` §「v11 文献」，ID 段 L01–L09 / B01–B05 / D01–D03 / N01–N03 / PRE01
- 代码：`reference_methods/{BALANS,DRHG,DORA,FLD4CO}/source` + 各自 `upstream.json`；`reference_methods/FLD4CO/operators/fld_sas.py`（129 行）、`reference_methods/LRBS/operators/lrbs_beam.py`（197 行）+ `LRBS/upstream.json`；`reference_methods/catalog.tsv` 追加 5 行
- 自检输出（可重跑）：`python reference_methods/LRBS/operators/lrbs_beam.py`、`python reference_methods/FLD4CO/operators/fld_sas.py`
