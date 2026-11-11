# RLNN 静态先验 v1 复盘与 v2 设计

> 日期：2026-11-10。分支 `codex/v7-one-shot-mamba`，HEAD `7667399`。
> 对象：[静态先验 v1 结果](RLNN_Static_Prior_v1_Results.md)、[v1 方案](RLNN_Static_Prior_v1_Analysis_and_Plan.md)、[Mamba v0 P0](RLNN_Mamba_v0_P0_Results.md)、[v7 总方案](RLNN_v7_OneShot_Mamba_Research_and_Implementation_Plan.md)。
> 现状：PAVE-Stats H1 保留；频域线已归档；Mamba 线的 v0 停在 P0，v1 换成廉价静态先验后未通过扩展准入。本文只分析 v1 为什么失败，以及 v2 怎么改。
> 本文的数值全部来自仓库已记录的结果，没有新实验。标注为"待测"的格子是 v2 才要填的。

---

## 0. 结论摘要

1. **v1 失败的主因不是"MLP 不如 Mamba"，而是注入接口没有杠杆。** 节点项每图内零均值、幅值上限 0.125，再乘 `tanh(alpha) = −0.15` 后上限 0.019 个 logit；实测总偏置 `delta_logit_rms = 0.0246`，平均绝对概率改变 **9.8 × 10⁻⁵**。这个量级作用在已饱和的 flip 概率上，等价于扰动 0.5% 的翻转决策。
2. **测不出差的另一个原因是目标量本身退化。** 四个臂、四个协议、共 8 次求解里 50 张图中有 **45 张大小完全相同**；不同臂选出的节点集合 Jaccard 只有 **0.44**（平均差 18.3 个节点）但**大小一样**。size 在这批图上是大片等值平台，微弱扰动只会换一个同样大的解。
3. **v1 的输入没有新信息。** 五维特征里 `density`、`size` 是图常量（只进常量通道），剩下三个节点变量中 `log_degree` 是 B 的显式输入通道，邻居度一二阶矩是 B 五层聚合可逼近的量。真正的全局结构（core、BFS 距离、两跳覆盖）被删掉了——而它才是 Mamba 命题的对象。
4. **训练目标把容量引向了翻转率，而不是解质量。** 训练损失里 λ·翻转数项（0.5 × 10.72）与能量项（4.22）同量级偏大；两个独立训练组收敛到几乎相同的常量（`c_add` −0.1317 / −0.1336，`c_drop` −0.0478 / −0.0485），方向都是"少翻转"，而少翻转在 0.5 秒下实测 **−0.10**（0 胜 90 平 10 负）。可辨识的部分全部落在全局常量上，节点项被判为无用并被 alpha 压小。
5. **v1 的判据把噪声当信号。** `equal_step` 门槛比较的是两次**独立训练**的臂（centered 对 constants，差 5 × 10⁻⁴ 个 logit），得出 +0.06；而唯一正确的同检查点干预（`masked`）在 T1000 与两档预算上给出 **maxabs = 0**（T1000 为 50/50，两档预算各 100/100 全同）。同时速度门槛在 T100 上测，而 T100 不是部署工作点，0.5 秒预算下这份一次性开销只占 0.5%、约 2 步。
6. **v2 必须先证伪接口，再谈编码器。** 用一个纯推理实验回答："任意静态节点信号，经每步 logit 偏置注入，能否在等步数下改变解大小？" 用振幅扫描、幅度匹配随机方向、以及 KaMIS 解构成的 oracle 上界三个对照。若 oracle 都没有杠杆，静态先验这条注入路线应当终止，不必再训练任何编码器。

---

## 1. v1 事实回顾

### 1.1 已完成的记录

| 项目 | v1 实际 | v7 方案要求 |
| --- | --- | --- |
| 训练图 / 更新次数 | 256 图 / 1 epoch / 512 次 | 1,000 图 / 3 epoch / 6,000 次 |
| 训练种子 × 参考种子 | 42 × ref0 | 42/123/999 × ref0/1/2 |
| 节点输入 | 5 维（度、邻居度均值/标准差、密度、规模） | 8 维（另加 core、BFS 距离、两跳覆盖） |
| 节点项幅值 | ≤ 0.125（乘 `tanh(alpha)` 后 ≤ 0.019） | `v7_delta_limit` 应为 0.5 |
| 筛选图 | 验证集第 51–100 张（50 张） | 100 张 |
| 每图搜索种子 | 1 个 | — |

### 1.2 等步数与预算结果

| 方法 | T100 A.3 | T100＋PAVE | T1000 A.3 | T1000＋PAVE | 0.5 秒 | 1.0 秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B | 19.62 | 19.74 | 20.14 | 20.14 | 20.14 | 20.16 |
| 双常量 | 19.54 | 19.76 | 20.06 | 20.06 | 20.04 | 20.14 |
| 中心化 MLP | 19.58 | 19.76 | 20.12 | 20.12 | 20.12 | 20.16 |
| 同权重去节点项 | 19.58 | 19.78 | 20.12 | 20.12 | 20.12 | 20.16 |

0.5 秒实际步数：B 983.7、双常量 1003.5、中心化 982.0、去节点项 983.8。1.0 秒：2416.5 / 2352.9 / 2396.1 / 2352.4。逐图记录见 `outputs/static_prior_v1/*_ref0/per_graph.csv`，汇总见 `outputs/static_prior_v1/analysis/summary_ref0.json`。

### 1.3 我复核出的三条关键事实

**(a) 同检查点干预严格为零。**

| 协议 | 中心化 − 去节点项 | 最大绝对差 |
| --- | --- | --- |
| T100 | +0.02（3 胜 2 负 45 平） | 1 |
| T1000 | 0.00（0 / 50 / 0） | 0 |
| 0.5 秒 | 0.00（0 / 100 / 0） | 0 |
| 1.0 秒 | 0.00（0 / 100 / 0） | 0 |

步数越少差一点，步数一多就完全一致。说明节点项只把早期轨迹推向另一个同样大的解，档案与 A.3 之后收敛到同一大小。

**(b) size 是大片等值平台，节点集合差得很多。**

0.5 秒下 B 与中心化的节点集合平均 Jaccard 0.44、平均对称差 18.3 个节点（最大 40），但大小只有 5/50 张图不同且都只差 1。四个臂 × 两次重复合起来，45/50 张图的大小完全一致。

**(c) 这批图在 0.5 秒已饱和。**

B 的 0.5 秒（984 步）与 T1000（1000 步）逐图大小 **50/50 完全相同**，1.0 秒只差 1 张。把四个协议、四个臂的最高值当作逐图可达上界，相对 B 的 0.5 秒只有 **+0.055**，且只有 6/50 张图能被任何一组改善。相比之下 T100 到 T1000 还有 **+0.40**（19/50 张图能继续改善），KaMIS 在 100 张验证图上比 RLNN 高 **+0.55**（[v3 交换设计](../已归档的想法和文件/RLNN_v3_Exchange_Search_Design.md)）。

换句话说：**v1 的筛选工作点（0.5 秒）已经贴着这套管线的天花板，而 v1 的固定工作量（T100）虽然有 0.40 的头空间，测的却是噪声。**

---

## 2. 逐层拆解：v1 到底失败在哪

下面按"影响从大到小"排列，每条都能独立地被下一轮实验否定或证实。

### 2.1 通道缺陷：注入量比决策边界小两个数量级

- 代码里两个常数取 `0.25 * tanh(c)`，节点项取 `0.125 * tanh(alpha) * centered(h)`。`--v7_delta_limit`（默认 0.5）在 `mis_static_prior.py::action_bias(topology, _limit)` 里被下划线参数**直接忽略**，所以振幅既不可扫也不可调。见 [models/mis_static_prior.py](../../models/mis_static_prior.py)。
- 训练后 `alpha = -0.1524`，`c_add = -0.1317`，`c_drop = -0.0478`。折算到 logit 上：加入偏置 −0.0327、移除偏置 −0.0119、节点项 RMS 0.0107。
- 反推斜率：`delta_logit_rms = 0.0246`、`delta_prob_mean_abs = 9.8e-5`，即平均 `σ'(z) ≈ 0.004`。只有在 `|z| ≳ 5` 的饱和区才会这么小——**B 的 logit 大多远离 0.5 决策边界**。
- 后果：一次 0.5 秒求解里翻转决策总数约 20 粒子 × 1000 步 × 5.6 次/步 ≈ 1.1 × 10⁵，其中受偏置影响的净增量约 0.5%。轨迹会分叉（Jaccard 0.44 就是证据），但落点大小不变。

**这解释了为什么"改变策略"和"改变大小"在 v1 里完全解耦。** 任何振幅 ≤0.125 的静态先验，在这批图上都不可能改变大小，与编码器是 MLP、MSG 还是 Mamba 无关。

### 2.2 信息缺陷：输入里没有 B 看不见的东西

五维特征的逐项归属：

| 特征 | 是否随节点变化 | B 是否已具备 |
| --- | --- | --- |
| `log_degree` | 是 | **是**，`MISStateGCN.state_features` 的第二个输入通道就是 `log1p(degree)` |
| `neighbor_log_degree_mean` | 是 | 五层聚合可逼近，另有邻居占用比例输入 |
| `neighbor_log_degree_std` | 是 | 同上 |
| `density` | **否**（图常量） | 只能进常量通道 |
| `size` | **否**（图常量） | 只能进常量通道 |

即：v1 的节点项本质上是一个"度数先验"，而度数已经是 B 的输入。真正带非局部信息的 core、BFS 距离、两跳覆盖被删掉了，它们在 v0 的代理探针里正是"八维 − 局部"那 +0.0078 AUC 的来源。

**这是 v1 最大的概念错误**：用一组已被骨干覆盖的输入去检验"全局结构先验是否有用"，得到的零结果只能否定"度数先验"，不能否定 Mamba 命题。

### 2.3 目标缺陷：λ 翻转数项主导，训练把常量推向有害方向

`solvers.erdoes_graph_batch_loss` 是 `expected_energy + λ·((E[flips] − num_d)²)`，v1 用 λ=0.5。训练末记录 `train_energy = 4.22`、`train_distance = 10.72`，第二项加权后 5.36，与能量项同量级偏大。

推理时 `num_t` 是固定的 1000 步，`num_d=5` 只是训练期的辅助正则，**不是部署目标**。但从梯度可辨识性看：

- 翻转数项对"全局抬高/压低 flip 概率"最敏感；能量项只对少数边界节点敏感。
- 两组独立训练收敛到几乎相同的常量，方向一致（都 < 0，即少翻转），说明可辨识信号确实来自翻转数项。
- 实测：少翻转在 0.5 秒下 **−0.10**（0 胜 90 平 10 负，100 个配对），在 T1000 下 −0.08（0 胜 46 平 4 负）。
- 节点项在每图内被中心化成均值为零，**恰好失去了唯一能影响翻转数的那个通道**，于是只剩 1e-4 量级的信息通道，被 alpha 压小（−0.15）。

也就是说：v1 的设计把"能动的"（均值）和"想验证的"（节点排序）拆开，结果能动的那部分被训练目标推到有害方向，想验证的那部分没有杠杆。

### 2.4 判据缺陷：把混沌当机制

| 问题 | 记录中的表现 |
| --- | --- |
| 用独立训练的臂做机制对照 | `centered` 对 `constants` 在 T1000 得 +0.06（3 胜 47 平 0 负），CI [0.00, 0.14]，据此判 `equal_step: true`。但两个臂只差 5×10⁻⁴ 个 logit，且同检查点的 `masked` 严格为 0。这个 +0.06 是混沌，不是机制 |
| 没有 null 带 | 全表没有"同策略换随机流"或"幅度匹配随机方向"的对照，无法区分 +0.06 与噪声 |
| 分辨率不足 | 50 张图、每图 1 个搜索种子，1 个节点 = 0.02；配对差的标准差约 0.3，SE ≈ 0.045，最小可辨效应约 ±0.09。而本轮所有效应都在 0.10 以内 |
| 速度门在工作点之外 | `speed` 门用 T100 整图时间比 ≤1.08；但 0.5 秒预算下同一份一次性准备只占 0.5%（12.35 → 14.82 ms），换来的步数损失约 2 步 |
| 没有剂量–反应 | 振幅是常量，扫不了，因此无法判断"是方向错了还是幅度不够" |
| 单训练种子 | 训练随机性（初始化、teacher 流、shuffle）完全没有重复 |

### 2.5 执行缺陷：两个 regime 混在一份记录里

| 记录 | 执行路径 | 0.5 秒实际步数 | 均值 |
| --- | --- | ---: | ---: |
| v1 `budget05` | FP32 ＋ CUDA Graph | 983.7 | 20.14（第 51–100 张） |
| v5 `0.5s×3 种子`（README 主数字 19.677） | FP16、eager（无 CUDA Graph） | 122.7–128.7 | 19.68（第 1–100 张） |
| v5 `1.0s×3 种子`（19.830） | 同上 | 247.1–287.2 | 19.83 |
| v6 固定 100 步 B | FP16 ＋ CUDA Graph | 100 | 46.7 ms/图 |

同一句"0.5 秒预算"在不同版本里意味着 128 步或 984 步，相差 7.7 倍；速度门槛（T100 1.08×）是在有 Graph 的 regime 里设的，而部署数字是在没有 Graph 的 regime 里测的。**v2 必须先声明执行 regime，并在同一 regime 内重测 B、重设门槛。** 这不影响 v1 的"节点项为零"结论（那是同 regime 内配对），但影响所有绝对值和预算结论。

### 2.6 范围缺陷：v1 其实没有测 Mamba

v1 从头到尾没有 Mamba 前向，`mamba_version` 只是 provenance 字段。所以 v1 的负结果对 Mamba 命题**没有证据力**，它只否定了一个由度数矩构成的 133 参数节点先验。v0 的 P0 也只测了正确性、速度和代理 AUC，没有测真实 MIS 质量。

---

## 3. 现在真正知道什么、不知道什么

**已经知道（证据充分）**

1. 在这批 RB200–300 验证图上，A.3＋冻结 PAVE 的管线在 0.5 秒（约 980 步）已饱和：继续加到 1.0 秒只 +0.02，逐图 49/50 不变。
2. 大小是平台型指标：不同策略可以给出 Jaccard 0.44、差 18 个节点的解，而大小完全相同。
3. ≤0.125 幅值的静态节点偏置对大小的作用严格为零（同检查点干预 maxabs=0）。
4. 全局"少翻转"是**有害**的（0.5 秒 −0.10，T1000 −0.08），因此任何后续先验都必须守住 B 的翻转率量级。
5. 可用头空间：T100→饱和 +0.40（19/50 图可改善）；KaMIS 参考 +0.55；本轮四个臂 × 四协议的并集上界只有 +0.055。

**还不知道（v2 要回答）**

1. 是否存在**任意**静态节点信号能在等步数下改变大小？（接口可达性）
2. 如果存在，需要多大振幅、作用在哪些节点？（剂量–反应与响应区间）
3. 提升是"收敛更快"（T100 曲线左移）还是"极限更高"（饱和值上升）？
4. 全局结构信息（core/BFS/两跳）相对纯度数先验是否有增量？Mamba 相对同信息 MLP/MSG 是否有增量？
5. 部署 budget 下的步数损失能否压到 2% 以内？

---

## 4. v2 设计

### 4.1 主张

v2 把 v1 混在一起的两个问题拆成两个可独立否决的命题：

- **P1（接口）**：存在某个静态节点信号 `b_i ∈ R²`、振幅 `A`，经 `ℓ = ℓ_B + A·b_{action(i)}` 注入后，在**同图同搜索种子同随机流**的配对下把等步数解大小推过 null 带。
- **P2（编码器）**：在 P1 成立的前提下，Mamba 的序列化静态编码优于同输入、参数匹配的 MLP/MSG。

v2 的名字只在 P2 成立后才有意义。P1 不成立就归档。

### 4.2 前提修正（先做，不花 GPU）

1. **锁定 regime**：主口径 FP32 ＋ CUDA Graph（K20，20 粒子）。同时保留一份 FP16 eager 对照并明确标注，因为历史部署数字在后者。所有速度/预算判据在同一 regime 内比较。
2. **图集**：主筛选用验证集第 **1–100** 张（与历史 100 图记录可比，且比第 51–100 张更难、头空间更大）；第 51–100 张作为已用过的次组，只在确认阶段复查。
3. **原语**：B、PAVE H1、A.3、CUDA Graph 捕获全部沿用；不改 `models/models.py`、`mis_v3_search.py`、`mis_v3_value.py`。
4. **固定对照**：所有臂共用同一张图的同一 `torch.Generator(seed)` 流（v1 已是如此，`capture_mis` 把 RNG 注册进 Graph，逐 step 的均匀随机矩阵形状一致），因此配对差只来自 logit。v2 要把这条写进 run.json 并在汇总脚本里断言。

### 4.3 P0：接口可达性判决（无训练，两卡并行约 1 小时）

**臂**

| 臂 | 内容 | 回答的问题 |
| --- | --- | --- |
| `b0` | 原 B，每图 3 个搜索种子 | 参照，同时给出采样噪声下界 |
| `null-rand` | 每图固定随机方向节点向量，RMS 与训练后 `static_bias_std` 相同（0.0107），并按图内均值归零 | 幅度匹配的"无效方向"（null 带的另一半） |
| `amp@0.125/0.25/0.5/1/2` | v1 中心化检查点，节点项上限改为这些值 | 剂量–反应，是否只是幅度不够 |
| `oracle-kamis` | 用 KaMIS 在验证图上的解构造偏置：解内节点 `b_add=+A`，解外节点 `b_drop=−A` | **完美静态先验的上界** |
| `oracle-B` | 用 B 的多次随机起点最大解的并集构造偏置 | 更宽松的上界（同一族内的最优方向） |
| `oracle-anti` | 用 `oracle-kamis` 的相反符号 | 方向性检验：应当变差 |

**协议**

- 收敛曲线：`T ∈ {25, 50, 100, 200, 400, 800}`，每点末步 A.3 解码（与 v1 的 T100 口径一致），另加每 10 步解码归档的口径。
- 两档预算 0.5 / 1.0 秒，`--post pave --extended-search`，预留 0.05 秒。
- 100 张图 × 每图 3 个搜索种子 × 2 次交错重复（第二次反转图内臂序）。所有臂共用同一组 `(图, 种子)` 流，`b0` 的三个种子之差直接给出采样噪声，不需要额外的空臂。

**指标**

- 主：Mean Size（A.3 与 ＋PAVE 分开报）、逐图 win/tie/loss、图级配对 bootstrap 95% CI。
- 副：`steps_to_target`（每图首次达到自身 T800 大小所需的步数）、轨迹分歧率（同 CRN 下逐节点翻转分歧率与最终集合 Jaccard）、边界节点比例（`0.01 < σ(z) < 0.99` 的节点占比）、翻转率（每步期望翻转数）、偏置 RMS 与 `mean|Δp|`。

**判据**

定义 null 带 `NB = max(同图换种子的 |Δsize|, |null-rand − b0|)` 的 95% CI 半宽（同图集、同协议）。

- P0 通过：存在 `amp` 或 oracle 使 `(arm − b0)` 的 CI 下界 **> NB**，且方向在两次重复里一致。
- P0 否决：所有 `amp ≤ 2` 与两个 oracle 都落在 NB 内 → 接口死刑，直接归档，不进入 P1。
- 只有 `oracle` 通过而 `amp` 全平 → 说明方向信息存在但当前学习方式学不到，P1 改为"对齐 oracle 方向"，不做编码器对比。
- 只有大振幅通过 → 取该振幅，并预期响应区间在 `|δz| ≈ 1` 附近（因为实测平均 `σ' ≈ 0.004`，B 的 logit 大多饱和）。若剂量曲线在 0.125–2 全程平坦，同样判否决。

**预测（写下来以便被否定）**

- `amp ≤ 0.5` 全程落在 NB 内，与 v1 一致。
- `oracle-kamis` 在 T100/T200 有正向效应（可能 +0.1 到 +0.3），在 0.5 秒被饱和吸收到 ~0。
- 同图换种子的 CI 半宽与 `null-rand` 的 CI 半宽都在 0.03–0.06 之间（50 图量级），提示需要 100 图 × 3 种子才能把 SE 压到 0.02 以下。

### 4.4 P1：训练协议修正（仅当 P0 通过）

| 项目 | v1 | v2 |
| --- | --- | --- |
| 输入 | 5 维 | 8 维（恢复 core、BFS 距离、两跳覆盖），并做逐特征消融 |
| 振幅 | 硬编码 0.25/0.125 | `--static_constant_limit` / `--static_node_limit`，取 P0 的响应值 |
| 目标 | `energy + 0.5·flip_penalty` | **λ = 0**（或 ≤0.05），只优化能量；把翻转率约束改为**报告项**，要求与 B 的 5.6 同量级 |
| 状态 | 冻结 teacher 状态（off-policy） | teacher ＋ 20% 学生 on-policy 状态（每 N 次更新用当前策略重跑一次采样） |
| 数据 | 256 图、512 次更新 | 1,000 图、3 epoch × 2,000 = 6,000 次更新（v1 实测 512 次 9 秒，6,000 次约 2 分钟，成本不是约束） |
| 种子 | 42 × ref0 | 42/123/999 × ref0/1/2 |
| 对照 | 独立训练的 constants | **同检查点** `amp=0`（节点项关闭）、**同 RMS 随机方向**、双常量、5 维 vs 8 维 |

训练期必须记录：`alpha`、`bias_rms`、`mean|Δp|`、`boundary_fraction`、`flip_rate`（与 B 对比）、`轨迹分歧率`（同 CRN 下教师与学生的状态差异，在第 25/50/100 步取样）、每 epoch 在固定 100 图上的 T100/T400/T800 size。若 `alpha` 继续被压向 0 且偏置落在 NB 内，如实判 No-go，不要靠加层数或加损失去救。

### 4.5 P2：编码器（仅当 P1 的正向效应超过 NB）

| 臂 | 输入 | 结构 | 识别问题 |
| --- | --- | --- | --- |
| `mlp8` | 8 维 | 参数匹配 MLP ＋ 图均值上下文 | 无序全局统计是否够 |
| `msg8` | 8 维 | 一次性两跳消息编码 | 额外传播是否够 |
| `mamba8` | 8 维 | 1 层共享双向 Mamba（width 16/32，d_state 16） | 序列化的边际贡献 |
| `mamba8-shuffle` | 8 维 | 同 Mamba，全随机序，独立训练 | 结构序是否必要 |
| `mamba8-degree` | 8 维 | 同 Mamba，degree 序 | BFS 序相对静态度序 |

速度门槛重写：**0.5 秒预算内的步数损失 ≤ 2%，P95 求解时间 ≤ 上限 ＋ 5 ms**；同时报 T100 整图比作参考但不作硬门槛。一次性准备成本目标 ≤ 2 ms（现有 4.3–8.0 ms）。

### 4.6 判据表（替代 v1 的 gates）

| 维度 | v2 判据 | 说明 |
| --- | --- | --- |
| 正确性 | 100% 可行，`amp=0` 时逐位等于 B，同检查点干预可复现 | 沿用 v1 的 check |
| 分辨率 | 图级配对差 SE ≤ 0.02（约 100 图 × 3 搜索种子） | 由 P0 的 NB 反推图数 |
| 接口 | 存在臂使 CI 下界 > NB | P0 |
| 机制 | **同检查点** `amp=0` 与随机方向的配对差都 ≤ NB，而实验臂 > NB | 取代 v1 的"centered vs constants" |
| 等步数 | T100 与 T400 曲线同时右移（CI 下界 > NB） | 用曲线而不是单点 |
| 预算 | 0.5 / 1.0 秒的 CI 下界 > NB | 与 B 同 regime |
| 速度 | 步数损失 ≤ 2%，P95 ≤ 预算 ＋ 5 ms | 取代 T100 1.08× |
| 编码器 | mamba8 相对 mlp8 与 msg8 的 CI 下界 > 0 | 仅在接口通过后 |

绝对阈值（19.62/19.70/19.85）**不要跨 regime 复用**：它们对应的是 FP16＋eager 或 FP16＋Graph 的旧口径，v2 必须在自己的 regime 里重测 B 再定阈值。

### 4.7 决策树

```text
P0 接口判决
├─ 全平（含 oracle）───────────► 归档：静态节点偏置路线 No-go
│                                部署保持 B ＋ 冻结 PAVE
│                                若仍想做 +0.55 级提升，改做盆地逃逸（2→3 交换）
├─ 仅大振幅有效 ───────────────► P1（用该振幅，8 维输入，λ=0，6000 更新）
│                                 └─ 同检查点干预 > NB ──► P2 编码器
└─ oracle 有效、学习臂无效 ────► P1'（先对齐 oracle 方向，做方向学习诊断）
                                  └─ 学到了 ──► P2
```

---

## 5. models/ 应该怎么改

目标：把"振幅"变成可扫的、把"偏置来源"变成可插拔的、把"机制对照"变成同检查点的、把静态特征的成本降下来。**不动骨干 GCN、不动 PAVE。**

### 5.1 文件与职责

| 文件 | 现状问题 | v2 改动 |
| --- | --- | --- |
| `models/mis_static_prior.py` | `action_bias(topology, _limit)` 忽略 `limit`；振幅硬编码 0.25/0.125；只有 constants/centered 两种模式 | 接收并用 `node_limit`、`constant_limit`；新增 `static_random`、`static_oracle`；新增 `bias_stats()` 返回 RMS 与均值 |
| `models/main.py` | 只有 `--v7_delta_limit`，对 static 模式无效 | 新增 `--static_constant_limit`、`--static_node_limit`、`--static_bias_seed`、`--static_oracle_path`；`--v7_delta_limit` 保留为别名并**真正生效** |
| `models/mis_v7_context.py` | 静态模式把 `delta_limit` 丢掉 | `forward` 与 `action_bias` 统一走一个 `limits` 参数；缓存 bias 时记下所用 limits |
| `models/v7_structure.py` | `structure()` 里 `small_graph_features` 对 n≤512 走稠密矩阵乘、`core_numbers` 是 Python 双层循环、每个分量一次 BFS + Python 逐节点回溯 | 2 跳覆盖改稀疏 `adj@adj`；core 用向量化剥离或 networkx 一次；BFS 距离用 `scipy.sparse.csgraph.breadth_first_order` 后 numpy 累加；按 `(num_nodes, edge_hash, schema)` 缓存特征与序 |
| `models/eval_static_prior.py` | 臂固定为 4 个；无振幅、无 null、无 oracle；无曲线协议；每图 1 个种子 | 臂规格化 `名称[:模式][@振幅][:来源]`；新增 `--steps` 曲线与 `--seeds-per-graph`；记录 `bias_rms`、`flip_rate`、`boundary_fraction`、`steps_to_target` |
| `models/diagnose_static_leverage.py`（新） | — | 复用 `diagnose_v7_context.rollout()` 的 CRN 重放，注入 oracle/random/amp 偏置，产出 P0 判决 |
| `models/train_v7_context.py` | 只记训练损失项；状态恒为 teacher | 支持 `--student_mix`；每 epoch 记 `bias_rms`/`flip_rate`/`divergence`；把 `--lambd 0` 变成合法配置并在 run.json 记录 |
| `train_scripts/mamba/v2/`（新） | — | `check.sh`、`p0.sh`、`train.sh`、`eval.sh`、`summarize.py` |
| `train_scripts/mamba/v1/summarize.py` | `equal_step` 用独立训练臂；`speed` 用 T100 | v2 版改成同检查点干预 ＋ null 带；旧文件保留为历史记录 |

### 5.2 关键代码

**振幅可扫、来源可插拔**（`mis_static_prior.py`）

```python
class StaticPrior(nn.Module):
    def __init__(self, mode, constant_limit=0.25, node_limit=0.125):
        ...
        self.constant_limit, self.node_limit = constant_limit, node_limit
        self.centered = mode == "static_centered"
        self.random = mode == "static_random"      # 幅度匹配的无效方向
        self.oracle = mode == "static_oracle"      # 目标解构成的偏置
        if self.random:
            self.register_buffer("direction", torch.zeros(0, 2), persistent=True)
        if self.oracle:
            self.register_buffer("target", torch.zeros(0), persistent=True)

    def action_bias(self, topology, limits=None):
        constant_limit, node_limit = limits or (self.constant_limit, self.node_limit)
        constants = constant_limit * self.constants.tanh()[None, :]
        if not self.centered:
            return constants
        h = self.context(topology.features).tanh()
        centered = torch.cat([h[a:b] - h[a:b].mean(0)
                              for a, b in zip(topology.ptr, topology.ptr[1:]) if b > a])
        return constants + node_limit * centered
```

`limits` 由 `V7StaticContextGCN` 持有并在 `policy_topology()` 里传入，这样评估时改 `solver.static_limits` 就能做振幅扫描，不必重训。

**同检查点干预 + 随机方向对照**（`eval_static_prior.py`）

```python
ARMS = ("baseline", "constants", "centered", "masked", "random", "oracle", "amp")
# masked : 复用 centered 权重，node_limit = 0
# random : 复用 centered 权重，direction 为每图固定随机向量（同 RMS，图内均值为零）
# oracle : 用 --static_oracle_path 的目标解构造 ±A 偏置
```

关键约束：`masked`、`random`、`amp@x`、`centered` 必须**共用同一个 checkpoint**，否则又回到 v1 的独立训练对照。

**曲线协议**（`eval_static_prior.py`）

```python
PROTOCOLS = {"t25": (25, None, None), "t50": (50, None, None), "t100": (100, None, None),
             "t200": (200, None, None), "t400": (400, None, None), "t800": (800, None, None),
             "t1000": (1000, None, 10),
             "budget05": (10000, .5, 10), "budget10": (10000, 1., 10)}
```

`policy.num_t = steps` 的用法可以原样保留；`steps_to_target` 需要 `observe=` 回调，在每 10 步解一次码并记录首次达到阈值的步号。

**一次性静态特征加速**（`v7_structure.py`）

```python
@lru_cache(maxsize=4096)
def _cached(n, key, schema_version):
    ...

def structure_cached(edges_bytes, n, order_seed, order=None, views=1):
    key = hashlib.blake2b(edges_bytes, digest_size=16).hexdigest()
    return _cached(n, key, FEATURE_SCHEMA_VERSION)
```

注意：缓存只用于**同一张图重复求解**（离线评测、诊断），部署口径仍要报"在线一次性"的耗时；两种口径分开记，不能用一个掩盖另一个。

### 5.3 不要改的东西

- `models/models.py` 的 GCN 数学定义、5 层结构、128 宽度。
- `models/mis_v3_search.py`、`models/mis_v3_value.py`、三份 PAVE 权重。
- `capture_mis.py` 的捕获方式：bias 必须在 `captured_states()` 之前算好并缓存，Graph 回放里不能出现编码器算子。
- A.3 解码规则、能量函数、`num_d=5` 的训练约束本身（v2 只是把它的**权重**调到 0，不改公式）。

### 5.4 工程检查清单

- [ ] `amp=0` 时逐位等于 B（FP32 同执行模式）。
- [ ] `masked`、`random`、`amp@x`、`centered` 共用同一 checkpoint，且 run.json 记录 checkpoint hash。
- [ ] 每个臂记录 `bias_rms`、`mean|Δp|`、`boundary_fraction`、`flip_rate`。
- [ ] 所有臂共用同一 `torch.Generator` 流；汇总脚本断言逐图 `search_seed` 相同。
- [ ] oracle 偏置只用验证图的目标解，测试集不得参与。
- [ ] CUDA Graph 回放中 selective scan / `context` 调用次数为 0。
- [ ] 静态特征与序的缓存 key 含 `schema_version` 与 `feature_norm_sha256`。
- [ ] 预算协议里 `policy_prepare_s` 单独记录，且分母是完整预算。

---

## 6. 实验怎么做

### 6.1 矩阵

| 阶段 | 组数 | 每组工作量 | GPU 时间（估） |
| --- | --- | --- | --- |
| P0 曲线 | 7 臂 × 6 个 T + 2 档预算 | 100 图 × 3 种子 × 2 重复 | 2 卡各约 60 分钟 |
| P1 训练 | 3 种子 × 4 臂（8 维、5 维、amp=0、随机方向） | 6,000 次更新 ≈ 2 分钟/臂 | < 1 小时 |
| P1 评测 | 4 臂 × 3 种子 × 4 协议 | 同上 | 2 卡各 2–3 小时 |
| P2 训练 + 评测 | 5 臂 × 3 种子 | 同上，Mamba 训练更慢 | 2 卡各半天 |

### 6.2 命令骨架

```bash
export RLNN_SOURCE_SHA="$(git rev-parse HEAD)"
# P0：无需训练，直接用 v1 的 centered 检查点做振幅扫描与 null/oracle 对照
CUDA_VISIBLE_DEVICES=0 bash train_scripts/mamba/v2/p0.sh curve   --gpu 0   # T25..T800
CUDA_VISIBLE_DEVICES=1 bash train_scripts/mamba/v2/p0.sh budget  --gpu 1   # 0.5/1.0 秒
python models/diagnose_static_leverage.py --ref 0 --samples 100 \
  --output outputs/mamba_v2/analysis/leverage_ref0
python train_scripts/mamba/v2/summarize.py --ref 0   # 计算 NB 并输出 P0 判决
```

P1/P2 沿用 v1 的 `train.sh`/`eval_matrix.sh` 结构，只替换 `--context_mode`、`--static_node_limit`、`--lambd`、`--student_mix`。

### 6.3 记录格式

每个 run 至少包含：`source_sha`、`backbone/adapters/PAVE` 三个 hash、`feature_norm_sha256`、`backend`（precision ＋ CUDA Graph）、`regime_note`、全部 limits、`graph_ids`、`search_seeds`、逐图的 `base_size/final_size/search_steps/search_s/prepare_s/pave_s/feasible/selected_nodes`，外加 `bias_rms`、`flip_rate`、`boundary_fraction`。汇总 JSON 里必须能直接读出 `null_band` 与每个臂对 B 的 CI。

### 6.4 统计口径

- 单位是"图 × 搜索种子"，先在每图内对重复取平均，再按图重采样（v1 已经这样做了，保留）。
- 报告 win/tie/loss 与 CI 下界，不用"最高种子"报成绩。
- 任何"机制"结论必须来自同检查点干预；跨训练组的差只作为工程结果。
- 先做 P0 得到 NB，再据此定图数（NB ≈ 0.05 时，50 图不足，需 ~100 图 × 3 种子把 SE 压到 0.02 以下）。

---

## 7. 风险与替代路线

1. **最可能的结局仍是 No-go。** 依据：T100 有 0.40 头空间，但四个臂在 T100 与 B 的差都在 ±0.08 内且配对 CI 全部跨零；而 B 的 logit 大面积饱和，任何"小偏置"都进不了响应区。若 P0 显示 oracle 都无效，这条路线应当结束，而不是继续调振幅。
2. **静态先验即使有效，也可能只在低预算区有效。** 0.5 秒已饱和，它的价值会体现为"更少步数达到同样大小"。这在 0.1–0.2 秒预算或更难的图上才有意义；如果要保留这条线，应该同时给出 T50/T100 的曲线，而不是只报 0.5 秒。
3. **真正的 +0.55 头空间在邻域之外。** v3 记录 KaMIS 改善的 52 张图里 48 张没有当前后备动作，说明瓶颈是盆地逃逸而不是局部排序。静态先验是"选盆地"的工具，如果要做，注入点应改成**能选择盆地的位置**（初始状态分布、A.3 的节点优先序、2→3 交换的候选排序），而不是每步 logit 上的微扰。这是 v2 之后的备选，不进本轮主实验。
4. **不要做的事**：不加大 Mamba 层数/宽度去救；不在每步调用 selective scan；不改 PAVE；不用作者 1,000 图调参；不在独立训练的臂之间下机制结论；不把 T100 速度比当部署门槛。

---

## 8. 需要同步更新的文档

- 本文落地后，在 [README](../../README.md) 的版本表里加一行 v2（结论待填），并注明 v1 的结论边界："5 维度数先验 ＋ ≤0.125 振幅 ＋ 512 次更新 ＋ 50 图单种子"。
- [v1 结果](RLNN_Static_Prior_v1_Results.md) 的"结论范围"段建议补一句执行 regime（FP32 ＋ CUDA Graph，0.5 秒约 984 步），避免与 v5 的 128 步口径混淆。
- `docs/实验结果.xlsx` 的 v2 工作表在每次实验后由 `tools/` 更新，规则不变。

---

## 附：一页纸摘要

- **v1 失败不是因为 MLP 弱，而是因为注入通道的杠杆被设计掉了**：节点项 ≤0.125 再乘 `tanh(alpha)`，实测 `mean|Δp| = 9.8e-5`；同检查点关闭节点项后 250 个配对大小全部相同。
- **size 在目标图上是大片平台**：45/50 张图四个臂大小一致，而节点集合 Jaccard 只有 0.44。扰动改变的是"哪一个同样大的解"，不是"多大"。
- **v1 的输入已被骨干覆盖**：五维里两维是图常量，`log_degree` 是 B 的显式输入，邻居度矩可逼近；core/BFS/两跳被删，而它们才是 Mamba 命题的对象。
- **训练目标把容量引向翻转率**：λ=0.5 的翻转数项与能量项同量级偏大，两个独立训练组收敛到相同的"少翻转"常量，而少翻转实测有害（0.5 秒 −0.10）。
- **判据把混沌当机制**：`equal_step` 用两次独立训练比较，得 +0.06 并判通过；同检查点干预严格为 0。速度门又测在非部署工作点。
- **v2 = 先判决接口，再谈编码器**：振幅扫描 ＋ 幅度匹配随机方向 ＋ KaMIS oracle 上界，无训练即可回答"有没有杠杆"；有杠杆才进入 λ=0、8 维输入、6,000 次更新、3 种子的训练修正；最后才是 MLP/MSG/Mamba 的编码器对照。
- **判据换成"超过 null 带"**，速度为"0.5 秒步数损失 ≤2%"，图数为 100 图 × 3 搜索种子。
