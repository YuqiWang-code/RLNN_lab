# RLNN_lab v4 机制追诊复盘与 v5 图频率模块重设计

> **文档用途**：供研究者 / Cursor / Codex 直接实施的技术方案（**未修改 GitHub，未执行 v5 训练**）。
>
> **对应任务**：RB200–300 最大独立集（MIS）；RLNN 负责动态二值状态搜索；**PAVE-Stats H1 保持冻结**，继续作为已经验证的后处理方法。
>
> **源码冻结点**：[`eaf290a239a87426106d23291a72d1852d1ed8a7`](https://github.com/YuqiWang-code/RLNN_lab/tree/eaf290a239a87426106d23291a72d1852d1ed8a7)，2026-11-09。最新提交标题为 `Record frequency follow-up results in workbook and snapshot`。若实际实施时 `main` 再更新，必须重新对比代码差异。
>
> **证据级别**：①“GitHub 已实测”表示从仓库日志、CSV、JSON 或报告核得的值；②“代码推导”表示对源码所定义运算的数学分析；③“v5 假设／设计”未经训练与测试，不能写成已获得的论文结果。

---

## 0. 先说研究决策（建议通读）

**结论不是“高频被 MLP 全部抹平”。** 最新 P1 复核修正了前次判断：F4 高频支路虽然原始输出的低频能量高达约 97.14%，但主要由**驻留（零频）分量**贡献；去驻留后，较高频段仍占 **90.009%**。说明至少在这批验证图上，高频内容仍存在，不能仅靠“加一个频率保持损失”来解释失败。

真正的失败链路是：

1. **特征已改变，但未证明决策更正确**。同状态改变翻转概率，并不自动提升多步 Bernoulli 搜索后的最终合法独立集大小。
2. **单节点短期收益本身很不稳定**。16 次重复下，半样本动作排序 Spearman 仍仅约 0.152；这是弱监督，不宜直接用它大规模训练频率头。
3. **频率双分支缺少独立价值**。在同信息、同参数的消息编码对照下，F4 固定步数的 Size 没有优势。
4. **搜索预算被吃掉**。F4 相比原 RLNN，训练图独占性能分析的 K20/T100 A.3 总时间约 **1.88 倍**；限时测试只完成约 59 步，对照约 98 步。

**建议设计一个轻量的 v5 候选：DAFR-Lite（Decision-Aligned Frequency Residual，决策对齐的图频输出残差）**。不是再在 GCN 输入端堆 128 通道的高低频编码，而是：

- 使用 MIS 的**当前节点选择与精确单点能量差**构造动态信号；
- 先以小 MLP 产生**节点标量修正提案**，再进行图低通 / 互补高通投影；
- **把投影后的标量修正加到原 RLNN 最终翻转 logits**，不穿越五层 GCN；
- 每个搜索步只新增至多两次稀疏邻居聚合（一次算选中邻居数，一次联合计算两路投影），不重复构建多尺度特征图；
- 用 `v5a_message`、`v5a_mlp`、解析能量增益对照证明频域参数化带来的**额外价值**；
- **PAVE 不参与频率模型训练**，仅在后期以同权重、同后处理协议进行验证。

如果早期实验显示频率与普通消息仍无区别或耗时无法接受，立即停止当前设计；不以增加频带、训练更久、增强 PAVE 的收益掩盖结构无效。

---

## 1. 本次重新核对了哪些文件

### 1.1 一手实验材料

| 文件 | 作用 |
|---|---|
| [`docs/已归档的想法和文件/RLNN_v4_Frequency_Mechanism_Followup_Results.md`](https://github.com/YuqiWang-code/RLNN_lab/blob/eaf290a239a87426106d23291a72d1852d1ed8a7/docs/已归档的想法和文件/RLNN_v4_Frequency_Mechanism_Followup_Results.md) | P1—P4 全部新诊断的结果与边界 |
| [`outputs/v4_frequency/followup/analysis/evidence_matrix.md`](https://github.com/YuqiWang-code/RLNN_lab/blob/eaf290a239a87426106d23291a72d1852d1ed8a7/outputs/v4_frequency/followup/analysis/evidence_matrix.md) | 各科研假设支持／不支持／不确定的证据矩阵 |
| [`outputs/v4_frequency/followup/analysis/decision.json`](https://github.com/YuqiWang-code/RLNN_lab/blob/eaf290a239a87426106d23291a72d1852d1ed8a7/outputs/v4_frequency/followup/analysis/decision.json) | `no_go / current_f4_dual_branch` 的正式阶段判断 |
| [`outputs/v4_frequency/followup/analysis/`](https://github.com/YuqiWang-code/RLNN_lab/tree/eaf290a239a87426106d23291a72d1852d1ed8a7/outputs/v4_frequency/followup/analysis) | 逐图配对、分阶段谱、统计置信区间、评分可靠性、时延分解与原始图表 |
| [`outputs/v4_frequency/summary.txt`](https://github.com/YuqiWang-code/RLNN_lab/blob/eaf290a239a87426106d23291a72d1852d1ed8a7/outputs/v4_frequency/summary.txt) | v4-A 初筛的不同精度、PAVE 与固定时间质量 |
| [`docs/temporary/RLNN_v3_Results_Analysis.md`](https://github.com/YuqiWang-code/RLNN_lab/blob/eaf290a239a87426106d23291a72d1852d1ed8a7/docs/temporary/RLNN_v3_Results_Analysis.md) | 冻结 PAVE 的可靠历史成绩、动作排序训练及强规则对照 |

### 1.2 源码与执行脚本

- `models/archived_frequency/mis_v4_frequency.py`：F4 的频率分解、门控、128 通道残差注入；**v5 不应直接在这里覆盖旧结构**。
- `models/archived_frequency/diagnose_v4_spectrum.py` + `models/archived_frequency/frequency_diag_utils.py`：逐层、去驻留频谱、教师／自身轨迹比较；v5 复用频谱算子与记录口径。
- `models/archived_frequency/diagnose_v4_branch_replay.py`：共享初始状态与随机矩阵的分支干预；v5 复用 replay 测试模式。
- `models/archived_frequency/probe_v4_utility_followup.py`：分层动作、R=2/4/8/16 可靠性与成对收益；v5 继续用作**诊断**，不能未经验证直接改成损失。
- `models/archived_frequency/profile_v4_frequency.py`：独占 GPU 的整图端到端与微组件计时；v5 只增加统计项，不改变定义。
- `models/solvers.py`：`RLNN.__init__`、`policy_topology()`、`flip_logits()`、`generate_data()`、`sample_states()`、`train_graph_batches()`；v5 要接入的关键接口。
- `models/main.py`：`--frequency` 方法选择；增加 v5 模式。
- `models/eval_v3.py` / `models/archived_frequency/eval_v4_frequency.py`：PAVE 冻结及公平评测入口；建议复制封装为 v5 版本。
- `train_scripts/legacy/v4_frequency/` 和 `train_scripts/legacy/v4_frequency/followup/`：保留旧运行，v5 新建独立目录。

本报告基于 GitHub **已提交的结果**分析，无法查看未推送到 GitHub 的服务器临时权重或进程状态。

---

## 2. 四组实验究竟证明了什么？

### 2.1 P1：逐层谱分解——纠正“高频全部丢失”的判断

**GitHub 实测**：三种子、验证图、共同冻结教师轨迹；输入高频与编码过程的谱统计：

| 位置 | 原始较高频能量（λ∈[1,1.5)） | 驻留分量 | 去驻留后的较高频 |
|---|---:|---:|---:|
| 原始 high 输入 | 91.095% | ≈0% | 91.095% |
| Linear1 | 12.500% | 86.404% | 91.944% |
| SiLU | 11.744% | 87.142% | 91.383% |
| Linear2 | 3.208% | 96.488% | 91.195% |
| LayerNorm | 2.601% | 97.111% | **90.009%** |
| 门控与 α 后 | 2.687% | 96.975% | 88.755% |

**新知识**：与其说高频输出变“平滑”，不如说其总能量中出现了很强的零频／驻留分量。在连接图、采用惰性随机游走算子的情况下，逐节点通道偏置等操作会引入常数方向。相对能量是分数；驻留分量大时，非驻留高频的**相对份额**会变小，不代表所有非驻留信息被抹除。

本项目使用的 `P=(I+D^{-1}A)/2` 对无向图是可逆平衡链，驻留方向按其对应归一化拉普拉斯与度权内积定义。对非连通图，应逐连通分量处理零频；孤立节点不能简单套用正度节点的权重。

**建议推翻的旧说法**：“高频 MLP 导致 90% 高频完全损失”“只要移除偏置就能恢复最终 Size”。P1 的置零偏置实验没有显示非驻留谱质量因此变好。

**仍不知道**：有用动作的信息是否主要在高频非驻留分量中？单纯保留该分量能否改善最终解？P1 不支持直接宣称。

**对 v5 的直接启发**：不用为了数字漂亮而给每层加图频谱损失；改成**先非线性编码，再对标量决策修正作图低/高通投影**，使最后要加到 logits 的高通修正在算子意义上抑制驻留方向，并实测它是否有用。

### 2.2 P2：同状态／共享噪声分支干预——两个分支不是“完全抵消”的证据

**GitHub 实测**：100 验证图×3 种子，同初始状态、同逐步 Bernoulli 均匀随机数；实际轨迹可因为策略不同而分叉。

| 方法 | FP32 Size | 相对 F4 | FP16 Size | 相对 F4 |
|---|---:|---:|---:|---:|
| 原骨干 B | 19.5367 | +0.0067 | 19.5300 | +0.0533 |
| F4（完整） | 19.5300 | — | 19.4767 | — |
| F4 关低支路 | 19.5033 | −0.0267 | 19.5367 | +0.0600 |
| F4 关高支路 | 19.5300 | 0 | 19.5467 | +0.0700 |
| F4 关适配器 | 19.5100 | −0.0200 | 19.5567 | +0.0800 |
| F4 门控固定 0.5 | 19.4600 | −0.0700 | 19.4633 | −0.0133 |

FP32 的同状态一步期望能量下降（数值越负表示该步改善越大）在完整 F4 为 −0.0581，关闭高、关闭低、关闭两路分别为 −0.0302、−0.0043、+0.0248。该指标支持两路对**即时能量**有贡献，但不是最终 Size 的保证。

**判断**：

- “两路一定互相抵消”**证据不足**。完整 F4 在 FP32 的一步能量贡献近似叠加，而最终 Size 未同步提高。
- “门控完全没用”**不成立**。FP32 固定 0.5 后 Size −0.070；但此为已训权重的推理修改，不能等同重新训练。
- “FP16 禁用高路证明高路有害”**说得过强**。FP16 下禁用高路提升 0.070，但该现象存在数值敏感性；需与 FP32 和独立训练消融一起解释。

**数值波动的证据**：FP32 原生重复的 B、F4 搜索结果一致；FP16 重复中，B 与 F4 在 300 例中分别有 263、262 例在至少一个观测点轨迹变化，但最终 Size 仅 3、8 例有差异。故早期机制研究以 **FP32** 作主分析，部署必须另外测 FP16。

**对 v5 的启发**：不把低高频激活当成特征图直接插入五层 GCN；换到**最终 logits 的小残差**，并记录“一步能量 → 完整 T100 搜索 → A.3 → PAVE”各级指标，找出最先出现性能错配的位置。

### 2.3 P3：动作收益可靠性——最需要尊重的失败信号

**GitHub 实测**：训练图冻结原 RLNN 继续策略；共享随机数，R=2/4/8/16 续跑；两半排名 Spearman：

| H=20，重复次数 R | 有差异的状态／60 | 收益 SEM | 两半 Spearman | 共同非平局候选对同向率 |
|---|---:|---:|---:|---:|
| 2 | 26 | 0.06250 | 0.1743 | 59.44% |
| 4 | 33 | 0.05760 | 0.1342 | 70.87% |
| 8 | 38 | 0.05326 | 0.1850 | 61.97% |
| 16 | 39 | **0.04332** | **0.1523** | 56.05% |

R=16 时，H=10/20/40 的两半 Spearman 约为 0.1483 / 0.1523 / 0.0310。加长 H 并没有得到更稳定的候选排序。

| R=8，20图×3种子×3阶段，分层／均匀采样 | F4 排序准确率 | MSG 排序准确率 | MSG regret − F4 regret |
|---|---:|---:|---:|
| 教师状态，Uniform16 | 45.44% | 45.46% | −0.00139 |
| 教师状态，Stratified24 | 51.25% | 51.45% | +0.00069 |
| F4 自身状态，Stratified24 | 51.85% | 52.03% | −0.00069 |
| MSG 自身状态，Stratified24 | 52.67% | 52.80% | +0.00139 |

分层比均匀采样覆盖了更多有意义的节点类别，但 F4 对 MSG 的优势方向不一致。教师分层主组 60 个图—种子状态中，只有 4 个的 Top1 收益存在差异，2 个对 F4 有利、2 个不利。

**关键区分**：RLNN 实际一步按**多个节点的独立 Bernoulli 概率**联合翻转。上述探针评价的是**单节点强制翻转后、再由冻结 B 继续运行**的收益。它只是代理诊断，既不是 RLNN 训练目标，也不是整个 T100+PAVE 的实际回报。由此不能推导“RLNN logits 49% 表示模型分类准确率只有 49%”。

**对 v5 的启发**：第一轮**不要加入单节点 rollout 回归或 pairwise ranking 辅助损失**。先继续使用已经稳定复现的 RLNN `erdoes` 训练与最终 Size 选择权重；单点收益探针用来监测是否更接近动作质量。若要在后续引入任务对齐监督，先建立稳定的**整步多节点动作、完整档案改进、或稳定局部结构动作**标签，并与随机扰动对照。

### 2.4 P4：计算开销——属于确定的工程瓶颈

独占 RTX 5090、训练前20图、K20/T100、5 次重复（原生 FP16 端到端，含拓扑、搜索、A.3、合法性核对，不含加载/H2D）：

| 方法 | A.3 端到端均值 | P95 | +冻结 PAVE 的端到端均值 |
|---|---:|---:|---:|
| B 原骨干 | **376.78 ms** | 482.54 ms | 501.49 ms |
| 状态 MLP | 697.71 ms | 897.36 ms | 824.78 ms |
| 普通消息 MSG | 701.80 ms | 893.05 ms | 826.93 ms |
| **F4** | **709.38 ms** | **928.67 ms** | **835.15 ms** |

**P4 的细分**（F4，每图 100 步等价）：

- 完整 `state_bands` 约 **92.92 ms**；包括选中邻居、释放机会、双尺度图传播。
- 低／高路编码各约 **33.31 / 35.95 ms**；门控融合约 **35.22 ms**。
- 完整适配器约 **224.91 ms**，GCN 骨干约 **229.18 ms**。
- Bernoulli 更新加能量档案约 **32.71 ms**。

这些项有包含关系，**不能直接相加当端到端时间**。但说明额外开销分散在多个稀疏调用和大量小张量算子，而不是某一个可删的“大卷积”。MLP、MSG 与 F4 都慢，说明当前三种匹配对照都执行了大量共同统计路径，并不意味着 F4 滤波独自导致所有耗时。

**真实0.5秒验证**：原骨干+PAVE T100 上限 Size 19.6233、约98.12步；持续搜索 B+PAVE 19.6500、约129.03步；MSG+PAVE 19.5200、约63.43步；F4+PAVE **19.4800、约59.09步**。

**对 v5 的启发**：优先压到 1–2 个新增稀疏聚合／步，**在标量 logits 端做投影**，避免每步 128 维高低频编码和几次不必要的图传播。并用 `v5a_mlp` 的**真正廉价实现**作效率对照，不能为了参数匹配让它每步白算未使用的频带。

---

## 3. 一句话回答“v4 失败的根因”

**目前不存在被实验唯一定位的单一根因。** 较强证据支持：

1. **学习目标错位**：`erdoes` 优化单步软能量与期望翻转数，不能保证有限时间内 K 条轨迹的“最好合法解”更大；当前频率特征未产生可验证的额外动作价值。
2. **位置错位**：F4 在 `initial_project` 后注入通道向量，经过后续 5 层残差 GCN 与 `final_project` 才得到翻转 logit。输入端分频的用途被主干重新混合，不能直接控制最终输出的频率角色。
3. **复杂度错位**：为了改变一个标量 logit，F4 每步计算多个图信号和 128 通道低高频表示，增加的计算超过其现阶段的质量收益。
4. **对照证明不了创新性**：F4 的 `[P²Z,Z−PZ,PZ−P²Z]` 与普通 `[Z,PZ,P²Z]` 线性互相恢复；相同信息与容量对照没有得到改善。

**不支持的简单解释**：高频彻底消失、门控全局饱和、两路必然抵消、只要重新训练 50 epoch 就能变好、只改成 FP32 就能稳定提升。

决定 v5 的原则是：**先改变信息进入动作决策的方式和成本结构，后讨论多尺度特征容量**。

---

## 4. v5 的方法定义与创新边界

### 4.1 暂定工作名

**DAFR-Lite：Decision-Aligned Frequency Residual**（决策对齐的图频输出残差）。本名称只用于实验版本，不应在未经对照验证前作为论文中已成立的创新命名。

它解决的具体问题不是“恢复漂亮的频谱”，而是：

> 在固定图结构和动态二值 MIS 搜索状态下，能否用**低成本、输出端的互补图滤波校正**，比一般的局部统计 MLP / 普通图消息混合更好地指导 RLNN 多节点翻转，并在同一完整求解时间内产生更大的合法独立集？

### 4.2 与旧模型的结构性变化

| 维度 | F4 已有结构 | v5 DAFR-Lite 假设 |
|---|---|---|
| 处理信号 | `Z=[x,log(1+c)]` 的输入频带 | 当前 `x`、精确已选邻居数、度数、精确单点能量差 |
| 图分频发生位置 | **MLP 前** `P²Z`,`Z−PZ`,`PZ−P²Z` | **标量编码与节点门控后** `Pu_L`, `(I−P)u_H` |
| 分支容量 | 2×`MLP→128`、LayerNorm | 一条 `5→12→2` 小共享编码 + 2 路门控 |
| GCN 融合 | 加入 `initial_project` 后，经过五层 GCN | 原 GCN 完全不改，在**最后 logits** 上加小残差 |
| 图传播 | 选中邻居、释放计数、双尺度 P（多次） | 选中邻居 `Ax` 一次 + 两路标量 `P[u_L,u_H]` 一次 |
| 高路约束 | 高频输入经 MLP+节点门控后可产生驻留 | 对非线性后结果再次施加 `(I−P)`，抑制驻留输出 |
| 边界控制 | α 初值 0.01，可训练 | 精确零启动 + 有界小幅 `tanh` 系数 |
| PAVE | 冻结 | 仍冻结；仅在后期测同一个 PAVE |

### 4.3 架构总图（数据形状）

```text
图 G=(V,E),  当前 K 条候选状态 x ∈ {0,1}^[N,K]
          │
          ├────────────────── 原 RLNN 五层 GCN ──────────────────┐
          │                                                    │
          │                                               logits_base
          │                                                 [N,K]
          ▼                                                    │
稀疏邻居选中计数 c=A_binary x     缓存：A_binary、deg、GCN adj     │
          │                                                    │
          ▼                                                    │
计算精确单点能量变化 ΔE_i=(1-2x_i)(βc_i-1)                     │
          │                                                    │
          ▼                                                    │
   动态决策信号 z=[x, log(1+c), log(1+d), c/max(d,1), tanh(ΔE/5)]
          │ [N,K,5]                                         │
          ├──────────────┐                                    │
          ▼              ▼                                    │
     Tiny MLP         独立门控                                │
   5→12→2+tanh       5→2+Sigmoid                              │
      │ [N,K,2]      │ [N,K,2]                                 │
      └───── × ───────┘                                        │
             │  u_L,u_H 先按节点逐状态调制                     │
             ▼                                                │
      一次稀疏联合传播 P[u_L,u_H]                              │
         ┌─────────────┴─────────────┐                          │
         ▼                           ▼                          │
   LOW = P u_L                 HIGH = u_H − P u_H               │
   温和低通                      去驻留高通                        │
         │                           │                          │
         └──── 有界全局残差系数 αL,αH ────┐                   │
                                         ▼                   ▼
                              logits = logits_base + αL LOW + αH HIGH
                                         │
                                  sigmoid(logits)
                                         │
                             Bernoulli 随机翻转 + 能量档案
                                         │
                                原 A.3 解码得到合法 MIS
                                         │
                           可选：完全冻结 PAVE-Stats H1
```

**绝不把 K 条独立轨迹当成图的空间维度**。所有稀疏传播只在节点维 `N` 上进行；同图的 K 个候选互不交换信息；训练的多个图使用块对角拓扑。

---

## 5. 数学设计：分解和残差必须服务于动作

### 5.1 MIS 的精确局部边际量（不引入伪标签）

当前能量（无向边每条计一次）

\[
E(x)=\beta\sum_{\{i,j\}\in E}x_ix_j-\sum_i x_i,\quad \beta=1.02.
\]

令 `c_i=∑_{j∈N(i)}x_j`，翻转单个比特 `x_i→1-x_i` 的**精确**能量差：

\[
\Delta E_i=(1-2x_i)(\beta c_i-1).
\]

- `x_i=0,c_i=0`：加入该点可使能量下降 1。
- `x_i=0,c_i≥1`：单独加入会产生冲突，不能直接理解为最终解收益。
- `x_i=1,c_i≥1`：删除冲突点可使能量下降。
- `x_i=1,c_i=0`：删除孤立选中点会使能量上升 1。

**局部 ΔE 是解析可得的强先验，但它不是 PAVE 的有限交换收益，也不是多比特联合翻转的全部能量变化。** 解析增益控制必须作为独立对照。

### 5.2 图算子与频带

`A_binary` 是没有自环的原始二值邻接；`d_i=∑_j A_ij`。使用现有的惰性随机游走：

\[
P=\tfrac12(I+D^{-1}A),\quad L(U)=PU,\quad H(U)=(I-P)U.
\]

对孤立节点约定 `PU=U`。对每个形如 `[N,K,C]` 的张量，展开 `K·C` 为列，一次 sparse-dense matmul 后还原形状。

`P` 的图频响应为 `1−λ/2`，`I−P` 为 `λ/2`（这里 λ 是归一化拉普拉斯特征值）。两者是**软低通／软高通**，不是严格正交的理想频带，更不能称为二维 Haar 小波。

**后投影的理由**：对于无向非孤立图，平稳权重 `π_i ∝ d_i` 满足 `πᵀP=πᵀ`，所以 `πᵀ(I−P)U=0`。这让高路输出在图稳态加权内积下没有驻留分量。对于非连通图，该结论逐连通分量成立；数值实验应确认实现的边方向与归一化写法一致。

因此，节点门控必须发生在 `H` 投影**之前**；如果先高通再给每节点乘不同权重，会重新注入驻留分量。这里的结构约束是确切数学性质，**但不是最终 Size 会变好的证明**。

### 5.3 轻量动态信号与最终校正

对每个图内节点 i、轨迹 k：

\[
z_{ik}=\left[x_{ik},\log(1+c_{ik}),\log(1+d_i),\frac{c_{ik}}{\max(d_i,1)},\tanh(\Delta E_{ik}/5)\right].
\]

共享轻量编码 `u=tanh(W₂ SiLU(W₁ z))∈[-1,1]^2`，独立局部门控 `g=σ(W_g z)∈(0,1)^2`，令 `v=u⊙g`。

\[
\delta_L=P v_L,\quad \delta_H=(I-P)v_H.
\]

用**有界**系数（例如 `a_L=0.25 tanh θ_L`, `a_H=0.25 tanh θ_H`）校正：

\[
\ell_{ik}^{v5}=\ell_{ik}^{RLNN}+a_L\delta_{L,ik}+a_H\delta_{H,ik}.
\]

`θ_L=θ_H=0` 时与原 RLNN **完全一致**（精确零启动）。首个梯度步主要更新残差系数，之后分支获得梯度，这是正常现象，应该记录梯度启动时间。`0.25` 是实验超参数，不是理论最优值；应用 `tanh` 防止初期大幅改写已复现的 RLNN 决策。

**为何不直接给低高频附加“正交损失”？** 高低两路的输入经过不同学习函数，软滤波的不同输出不一定正交；目前没有由数据或收益标签支持强迫隐藏特征正交。额外正交约束可能损害任务相关信号。

### 5.4 v5 的新颖性不能夸大

图低频、高频、波形分解、门控都已有论文先例；另见 §15。**DAFR-Lite 的真正待验证贡献**是“动态 MIS 决策信号 × 非线性之后的图频输出端约束 × 低预算联合翻转搜索”的具体机制及其已证实的性能和效率。若不能超过匹配普通消息与局部统计版本，仍不构成成立的频域创新。

---

## 6. `models` 该怎么改：准确到文件与函数

**总原则**：原 v1—v4 模块、v3 PAVE、论文基线和测试记录均不覆盖。新建 v5 文件，通过 `--frequency` 分流。保证旧模型 loading 的 `state_dict` 键仍然兼容。

| 文件 | 操作 | 关键内容 |
|---|---|---|
| `models/archived_frequency/mis_v5_frequency.py` | **新增** | v5 Tiny DAFR、`V5_MODES`、拓扑/信号、输出端校正、频率与普通消息模式 |
| `models/main.py` | **小改** | 添加 `V5_MODES` 到频率模式选择 |
| `models/solvers.py` | **小改** | v5 构建、热启动兼容、`policy_topology()`，保留 `generate_data()/sample_states()` 旧逻辑 |
| `models/archived_frequency/train_v5_frequency.py` | **新增** | 同教师状态训练与 v5 详细日志；复用 v4 训练预算和最佳权重选择 |
| `models/archived_frequency/eval_v5_frequency.py` | **新增** | 复制 v4 评测封装，模式和路径改 v5，v3 PAVE 仍被冻结 |
| `models/archived_frequency/diagnose_v5_frequency.py` | **新增** | 轻量诊断与相同轨迹/随机数的收益比较；参考 v4 P1/P2 源码 |
| `models/archived_frequency/profile_v5_frequency.py` | **新增** | 以独占 GPU 拆分 `Ax`、Tiny MLP、联合 `P`、GCN、解码、PAVE 时间 |
| `train_scripts/legacy/v5_frequency/` | **新增** | `train.sh`、`eval.sh`、`eval_budget.sh`、`smoke.sh`、`diagnose.sh`、`README.md` |
| `outputs/v5_frequency/` | **新版本产物** | JSON/CSV，不把模型权重直接传入 Git |

### 6.1 新增 `models/archived_frequency/mis_v5_frequency.py` 的推荐初版源码

下面是按已确认的现有 `GCN`、`v4_topology()` 接口写的**可编译的实施参考代码**，不是已经合入仓库且训练成功的版本。形状为 `[N,K]`，支持块对角多图、FP32 状态统计和 GPU 自动混合精度；下一步必须先运行 §7 的单元测试再上服务器训练。

```python
"""v5 DAFR-Lite: lightweight post-nonlinearity graph-band logit residual."""
import torch
from torch import nn

from models import GCN
from archived_frequency.mis_v4_frequency import v4_topology

V5_MODES = (
    "v5a", "v5a_message", "v5a_mlp", "v5a_low", "v5a_high", "v5a_fixed",
    "v5a_noenergy",
)


def v5_topology(edge_index, energy_adj, nodes):
    # First iteration: deliberately reuse the validated v4 binary CSR topology.
    return v4_topology(edge_index, energy_adj, nodes)


def lazy_scalar(signals, topology):
    """[N,K,2] -> P@signals, propagating only along graph nodes."""
    assert signals.ndim == 3 and signals.size(-1) == 2
    n = signals.size(0)
    count = topology.degree.reshape(n, 1, 1)
    neighbors = (topology.binary @ signals.flatten(1)).reshape_as(signals)
    average = neighbors / count.clamp_min(1)
    average = torch.where(count > 0, average, signals)
    return 0.5 * (signals + average)


class V5DAFR(nn.Module):
    def __init__(self, mode="v5a", width=12, beta=1.02):
        super().__init__()
        assert mode in V5_MODES
        self.mode = mode
        self.beta = beta
        self.signal = nn.Sequential(nn.Linear(5, width), nn.SiLU(),
                                    nn.Linear(width, 2), nn.Tanh())
        self.gate = nn.Linear(5, 2)
        # Exact zero-initialized correction: safe warm-start from RLNN.
        self.theta_low = nn.Parameter(torch.zeros(()))
        self.theta_high = nn.Parameter(torch.zeros(()))

    def forward(self, x, topology, return_details=False):
        # All reductions use float32 even under outer FP16 autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            n, k = x.shape
            counts = topology.binary @ x
            degree = topology.degree.expand(-1, k)
            delta_e = (1 - 2 * x) * (self.beta * counts - 1)
            z = torch.stack((
                x,
                counts.log1p(),
                topology.log_degree.expand(-1, k),
                counts / degree.clamp_min(1),
                (delta_e / 5.0).tanh(),
            ), dim=-1)
            if self.mode == "v5a_noenergy":
                z = z.clone()
                z[..., 4] = 0.0
            proposal = self.signal(z)
            gates = self.gate(z).sigmoid()
            if self.mode == "v5a_fixed":
                gates = torch.full_like(gates, 0.5)
            v = proposal * gates

            if self.mode == "v5a_mlp":
                # Truly cheap non-spectral capacity control; no unused sparse P.
                low = v[..., 0]
                high = v[..., 1]
            else:
                propagated = lazy_scalar(v, topology)
                low = propagated[..., 0]
                if self.mode == "v5a_message":
                    # Same z, network parameters and P; standard local-message mix.
                    high = 0.5 * (v[..., 1] + propagated[..., 1])
                else:
                    # Apply high-pass AFTER nodewise nonlinear encoding and gating.
                    high = v[..., 1] - propagated[..., 1]

            if self.mode == "v5a_low":
                high = torch.zeros_like(high)
            if self.mode == "v5a_high":
                low = torch.zeros_like(low)
            a_low = 0.25 * self.theta_low.tanh()
            a_high = 0.25 * self.theta_high.tanh()
            correction = a_low * low + a_high * high
            if not return_details:
                return correction
            return correction, dict(
                z=z, counts=counts, gates=gates, proposal=proposal,
                low=low, high=high, a_low=a_low, a_high=a_high,
            )


class V5DAFRGCN(GCN):
    def __init__(self, hidden, layers, beta=1.02, mode="v5a"):
        super().__init__(1, hidden, layers, 1)
        # Use 'adapter' key to minimize changes to RLNN's freeze/warm-start code.
        self.adapter = V5DAFR(mode=mode, beta=beta)

    def forward(self, x, edge_index, energy_adj, topology=None):
        if topology is None:
            topology = v5_topology(edge_index, energy_adj, x.size(0))
        base_logits = super().forward(
            x.unsqueeze(-1), edge_index, topology.normalized,
        ).squeeze(-1)
        correction = self.adapter(x, topology)
        return base_logits + correction.to(base_logits.dtype)
```

**注意 1——上述实现中的 `v5a_message` 是同信息、同传播次数、同参数的**受限消息对照**，不是全表达能力无约束的普通消息基准。正式发表前还需加入一个更强的可学习系数控制，例如同时使用 `v` 和 `Pv` 的自由线性组合（额外参数应对齐），让频域模型面对更强的反证。

**注意 2——`v5a_low/high/fixed` 在这份源码里是训练模式，必须从同一起始骨干**独立训练**后才算严格消融。对完整模型的推理“置零”仍应单独标为干预。

**注意 3——Tiny 结构是起点而非最终设计。** 高路通过 `(I−P)` 约束具有零驻留分量，但 `sigmoid` 后翻转概率当然不再严格线性频域分解；需要实测，而非声称整条搜索策略“频率正交”。

### 6.2 `models/main.py` 的最小修改

在既有频域模式导入附近新增：

```python
from archived_frequency.mis_v5_frequency import V5_MODES
```

修改 `--frequency` 参数的 `choices`，**追加** `+ V5_MODES`，旧版模式原样保留：

```python
parser.add_argument(
    "--frequency",
    choices=("none", "stats", "low", "high", "fixed", "local", "dynamic")
            + SEGF_MODES + V4_MODES + V5_MODES,
    default="none",
)
```

### 6.3 `models/solvers.py` 的四处最小修改

1. 文件顶部导入：

```python
from archived_frequency.mis_v5_frequency import V5DAFRGCN, V5_MODES, v5_topology
```

2. `RLNN.__init__` 中已有 `self.segf` / `self.v4_frequency`，新增：

```python
self.v5_frequency = self.frequency in V5_MODES
if self.frequency != "none":
    if args.problem != "mis":
        raise ValueError("Graph-frequency encoders require MIS")
    if self.v5_frequency:
        self.model = V5DAFRGCN(args.num_h, args.num_l, args.beta, self.frequency)
    elif self.v4_frequency:
        self.model = V4FrequencyGCN(args.num_h, args.num_l, args.beta, self.frequency)
    elif self.segf:
        self.model = SEGF(args.num_h, args.num_l, args.beta, self.frequency)
    else:
        self.model = MISStateGCN(args.num_h, args.num_l, args.beta, self.frequency)
else:
    self.model = GCN(1, args.num_h, args.num_l, 1)
```

3. 热启动检查：原来只接受 `self.segf or self.v4_frequency` 的适配器缺失键，修改为：

```python
strict = not (self.segf or self.v4_frequency or self.v5_frequency)
missing, extra = self.model.load_state_dict(weights, strict=strict)
if extra or any(not name.startswith("adapter.") for name in missing):
    raise ValueError("Unexpected warm-start keys; refuse to train")
```

`warm_start` 必须是同种子**原结构 RLNN**，不可误用 v4 checkpoint 作为 v5 原始骨干。

4. `policy_topology()` 优先判断 v5。`flip_logits()` 不需要重写，因为 v5 与 v4 同样接受 `(x,edge_index,adj,topology)`：

```python
def policy_topology(self, graph, adj):
    if self.v5_frequency:
        return v5_topology(graph.edge_index, adj, graph.num_nodes)
    if self.v4_frequency:
        return v4_topology(graph.edge_index, adj, graph.num_nodes)
    if self.segf:
        return frequency_topology(graph.edge_index, adj, graph.num_nodes)
    return edge2adj(graph.edge_index, graph.num_nodes)
```

其余 `sample_states()` 的 `K=20,T=100` 和能量档案、`erdoes_graph_batch_loss` 的定义**保持不变**。这样后面才有资格说性能变化来自 v5 模型，而不是搜索预算或解码器修改。

### 6.4 重要的实现优化与公平性问题

- `v4_topology` 的 `binary` 是 float32 稀疏 CSR、`degree` [N,1]；对 FP16 输入先转 FP32 再计算信号，校正后转回 logits dtype。
- **不能缓存随 x 变化的 `counts`、`delta_e` 或频带**；只缓存静态图、度数和归一化邻接。
- 在 `train_graph_batches()` 中，块对角图的 `A_binary` 不会传播到别的图；`v5a` 不做 across-graph 均值，因此不需要 `graph.batch` 来定义全图门控。
- `zero alpha` 使所有模式在初始化和同骨干加载时具有**相同输出**。必须实测 `max_abs_logits_diff≈0`。如果不成立，先修 bug，不开启训练。
- `v5a_mlp` 应跳过未使用的第二次稀疏传播，用来测**真实便宜 MLP**。另外可增加一版“等传播成本 MLP”，单独研究频率的归纳偏置；不要混淆“参数公平”和“时间公平”。
- 当前 `GCN` 主干仍约165,505参数；上述 `V5DAFR` 在 `width=12` 时约112个新增参数（5→12→2共享编码98、5→2门控12、两残差标量2）。具体以运行时参数计数为准；相对 F4 的额外 5,008 参数小得多。
- α 最大幅值不等于整体概率改变量的严格上界；应实际记录 Δlogit、Δp 分位数，特别监测 `x=1,c≥1` 冲突点与 `x=0,c=0` 可加入点。
- 这版轻量设计依然要计算额外 `Ax`，不能声称已达到原 RLNN 的计算成本；效率以真实 P4 计时为准。

---

## 7. P0：正式实施前必须跑的单元测试

建议新增 `train_scripts/legacy/v5_frequency/check_adapter.py`，参考原 [`train_scripts/legacy/v4_frequency/check_adapter.py`](https://github.com/YuqiWang-code/RLNN_lab/blob/eaf290a239a87426106d23291a72d1852d1ed8a7/train_scripts/legacy/v4_frequency/check_adapter.py)。小图 CPU 与 RTX5090 CUDA 各测试一次。

| 测试 | 预期 |
|---|---|
| 空图/孤立节点 | `P(x)=x`，高通输出为零；没有 NaN/Inf |
| 常数输入 | `P(1)=1`，`(I−P)1=0` |
| 邻居计数 | `c` 与原 `2*(energy_adj @ x)` 精确或容差一致 |
| 数学分频 | 对随机 U，`Pu+(U−Pu)=U` |
| 稳态投影 | 对每个连通块、每条粒子，`sum_i d_i*(U−PU)_i≈0` |
| 重标号等变性 | 同一图节点置换后 logits 按相同置换变化 |
| K 轨迹独立性 | 粒子列置换后输出同列置换，K=1/20/64/1000 |
| 多图块对角 | 单图独立计算与拼图计算的结果、反向梯度接近 |
| 热启动 / 零残差 | 同一骨干权重、FP32 下 v5 的 logits 与原 GCN 一致 |
| FP16/FP32 | 混合精度推理有限、可以训练且不跨图串扰；确认 autocast 选择 |
| 梯度可达性 | 第一更新后 α 可改变，后续 `signal` 与 `gate` 也获得非零梯度 |
| PAVE 不变 | v3 PAVE checkpoint 的 SHA256 与部署前完全一致 |
| 参数约束 | v5a、v5a_message、v5a_mlp 新增参数一致，时间单独报告 |

**必须补测一项**：当高路恒定为常数（例如人为设 `u_H=3`）时，`H(u_H)` 数值接近零；这比仅查看“较高频比例”更直接检验后投影结构。

建议在 `models/archived_frequency/frequency_diag_utils.py` 中复用成熟谱计算；不要在 v5 部署路径执行 `torch.linalg.eigh()`，完整特征分解只用于小图诊断，不能计入在线模型能力。

---

## 8. P1：没有训练前的机制筛选（低成本）

### 8.1 目的

先验证三个问题：

1. 新模型初始等价于 RLNN，后投影确实**控制驻留**；
2. 哪类图信号及哪类节点使修正更可能对动作有帮助；
3. 两次额外稀疏聚合是否已经使计算预算不可接受。

### 8.2 样本和设置

- 图：先训练集20图用于开发，不用于模型选择；后验证集100图。
- 检查点：v2 `none_seed0/1/2` 对应 `best.pt`；对照 v4 F4/MSG 仅作为历史参照，不重训练 PAVE。
- 固定 K=20、T=100；观察 `t=0,10,25,50,75,100`；各图前4粒子用于细谱诊断。
- 轨迹：同源 B 教师状态 + 各模型自身状态，两种结果分开记录。可比较同状态 forward 的频谱与动作偏好；不可把不同学生状态误称“同状态”。
- 精度：FP32 为主，FP16 做单独复核；自带固定初始、每步统一 U 的独立 Bernoulli 采样控制。

### 8.3 必须输出的结果

`outputs/v5_frequency/diagnosis_p1/`：

- `reconstruction.csv`：`Pu + H(U) − U` 最大相对误差、零频保留比例；
- `stage_spectra.csv`：`z / proposal / gated_proposal / δL / δH / δ / logits_base / logits_final` 的驻留比例、去驻留谱、谱重心；
- `node_groups.csv`：按 `(x,c)` 分为冲突已选、可直接加入、唯一阻挡、多重阻挡等节点，统计 `Δlogit、Δp、ΔE`；
- `gate_summary.csv`：gL/gH 的均值、10分位、90分位与 t 阶段变化；
- `runtime_micro.csv`：`Ax`、tiny MLP、`P[2 scalar]`、总 forward 占用。

**不能以“高路去驻留高频占比更高”作为进入下一轮的唯一理由**。那是结构的预期数学性质；最终必须由解质量证明。

---

## 9. P2：最关键——公平短训与固定工作量实验

### 9.1 主实验 arm（最小足够矩阵）

| Arm | 说明 | 作用 |
|---|---|---|
| `B` | 冻结已复现 RLNN，原架构 | 原始性能下界与时间基准 |
| `B+FT` | 同预算继续微调原 RLNN | 排除额外优化预算带来的收益 |
| `v5a_mlp` | 同5维决策统计、同参数、**不额外 P** | 排除容量和 MIS 解析统计造成的收益，并测廉价强对照 |
| `v5a_message` | 相同 z、编码、门控与 P 调用，普通消息混合 | **判断“频率约束”相对普通消息是否有效** |
| `v5a` | DAFR-Lite，低通+高通的最终 logit 修正 | v5 主要候选 |
| `Analytic-ΔE` | 原 RLNN logits + 校准的精确单点能量增益 | “是否只要注入已知增益就够”的强非学习对照 |

其中 `v5a_mlp/message/v5a` 应完全共享初始 GCN 权重和训练数据/图顺序/随机状态，所有可训练臂的新增参数严格一致；`Analytic-ΔE` 结构不同，必须单列参数量和校准方式，不能硬称参数匹配。

更严格的消息对照建议另加 `v5a_free_mix`：采用同样的一次 P 调用，但自由学习 `U`、`PU` 的线性组合。该臂比频率结构略更灵活，**频率胜过它才是更有说服力的结构性证据**；增加的少量参数应报告或等量对齐。

### 9.2 训练设计

**A1 短筛**：三个训练种子 `0/1/2`，训练图1000、验证图100，**3 epoch、每轮2000更新，共6000 Adam 更新**。学习率、batch、优化器、状态来源对照现有 v4；优先采用已经验证过的“冻结 B 教师共同生成状态”，使各臂暴露于一致状态分布。

- 第1 epoch：只更新 `adapter`；需保证优化器中对骨干的 `requires_grad=False` 正确生效。
- 第2–3 epoch：解冻骨干，采用原文已有 `2e-5` 微调学习率，三种子一致。
- Teacher：每图K=20、T'=50，采样状态只来自训练图，教师不回传梯度。
- 优化：**只使用原 `erdoes` 目标**（软能量与期望翻转数正则），不偷偷添加 PAVE 监督/伪标签。
- 每轮验证：固定验证随机流 `1017+graph_idx`，用 A.3 的**合法 Size**选择 checkpoint；不能用 train loss 的下降代替验证决策。
- 保存 `alpha / gate分布 / Avg |Δp| / train_energy / train_distance / train_flips / val_Size / val_time / 训练更新数`。可沿用新版 `solvers.py` 已有 `train_energy/train_distance/train_flips` 记录。

**训练源差异**：v4 现有 `train_v4_frequency.py` 中 `MatchedStateRLNN` 恰好提供教师共同状态逻辑，v5 可参考其接口复制一个独立训练器。`models/solvers.py` 的 `train_graph_batches` 会在迭代中调用 `self.generate_data(...)`，因此无需重写主训练循环。

### 9.3 验证主表：FP32 固定 K/T，不接 PAVE

| 方法 | Val Size 三种子均值±SD | ΔB | ΔB+FT | ΔMSG | Time | 参数 | 可行率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B | 待实验 | — | — | — | 待实验 | 165,505 | 待实验 |
| B+FT | 待实验 | 待实验 | — | — | 待实验 | 165,505 | 待实验 |
| MLP | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |
| MSG | 待实验 | 待实验 | 待实验 | — | 待实验 | 待实验 | 待实验 |
| **DAFR-Lite** | **待实验** | **待实验** | **待实验** | **待实验** | **待实验** | **待实验** | **待实验** |
| Analytic-ΔE | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |

同图逐图记录配对胜/平/负，三种子训练结果均值样本 SD；区间以图 ID 为重采样单位，不能把 300 条图-种子记录当成独立的 300 张图。

**建议筛选标准（预先写入 `decision.json`，不是既成事实）**：

1. DAFR 相对 MSG、MLP 的**平均** Size 差均正，且至少2/3种子非负；
2. 相对 B+FT 不能只因 B+FT 意外退化而胜出；最好也不低于最初 B；
3. 同状态的高路稳态抑制数值正确，存在非零的动作偏好改变；
4. 额外时间不能抵消收益：至少在0.5秒/图下，DAFR+PAVE 不劣于原 B+PAVE 与廉价MLP，并对强持续搜索对照报告差距；
5. 如统计区间横跨零，应报告“不确定”，不能宣称稳定胜利。

若不满足 (1)–(2)，不应因为个别 Seed 或 FP16 结果好看就升级为论文主方法。**先做一次预先指定的局部特征替换试验**，再判断是否停止频域线。

### 9.4 FP16 独立验收

将已选权重使用 FP16 再运行固定100步测试；**不能把 FP32 与 FP16 同种子的差异直接解释为“精度造成的模型效应”**，因为浮点采样流、稀疏归约与轨迹分叉都变了。

检查：① 同一条件空闲卡上原生重复的 Size 差；② 同随机数流的 paired replay；③ fp16 下修正是否过大；④ 0.5 秒里每图实际完成步数分布及超时数。

---

## 10. P3：频率能否真的改善搜索决策？

建议在 `models/archived_frequency/diagnose_v5_frequency.py` 复用已有 v4 utility/replay，而不新增一个复杂的训练目标。

### 10.1 同状态的四层因果链（仅作算法干预，不称后门调整）

用相同图 `G`、相同二值状态 `x`、相同后续均匀随机数序列 U 比较：

1. **即时 logits/概率**：`Δlogit`、分节点类别的 `|Δp|`、Top5 翻转概率节点重叠。
2. **一步联合翻转目标**：同 U 产生两组实际多比特更新 `x'`，记录 `E(x')−E(x)`，冲突数、选中节点数、期望 flips；也可以按解析软目标求 `E[ΔE]`。
3. **完整搜索**：T100 的每条轨迹 best-energy、best-state 冲突、最终 A.3 Size；记录图级 K20 最大值。
4. **冻结 PAVE**：对两个模型产生的各自合法起点调用完全相同的 PAVE-Stats H1，报告 `PAVE gain` 与 final Size。

上述“共同随机数”降低比较方差，但不同模型可产生不同轨迹；**不能将两个模型的独立起点当成相同起点**。若要研究“给同一合法起点 PAVE 不同评分器”，这已经属于 PAVE 研究，不在本轮修改范围。

### 10.2 不再只报告原始高频占比

必须跟踪 `proposal→gated→P低路/(I−P)高路→logit_delta→sigmoid_delta`。高路去驻留本来是运算定义保证的，重点改测：

- 高路是否**在冲突已选节点 / 可加入节点 / 唯一阻挡节点**产生差异？
- 频率修正是否提高 **相同动作预算下的联合能量下降与冲突修复效率**？
- 在搜索后期 `t=75/100` 能否改变仍有改进空间的决策？
- 若 logit 比例已经很小，概率变化分布是否主要落在饱和区，导致真正翻转事件很少？
- 比 MSG/MLP 能否带来额外**图级 decoded Size**，还是只有可视化差异？

### 10.3 单节点收益探针仍有价值，但只作次要诊断

继承 v4 `Uniform16` 与 `Stratified24` 两种采样。默认至少 R=8；若两半 Spearman 仍很低，则标注弱标签，不加入训练。对 `x=0,c=0` 等可加入节点，可计算**精确单步能量优势**；但不能因此直接认定长程收益。

若要进一步研究 PAVE 所揭示的交换机会，建议记录同一图的“解码起点结构潜力”，例如合法起点的可执行严格 1→2 数量、经过 PAVE 规则搜索的继续增益。**这些量作为诊断标签，不回传 PAVE 权重，不修改 PAVE 决策**。

---

## 11. P4：效率实验、部署开销与轻量化决策

### 11.1 计时范围

与 v4 同一口径：图和数据提前加载到 GPU、同步后开始单图计时；包含**CPU图构造 + 策略拓扑 + 网络前向 + Bernoulli 更新 + 能量档案 + A.3 + PAVE/合法性检查**；不计 Python 进程启动、读图和初次 H2D。GPU 每卡同时只有一个正式评测任务，先预热，再交错重复。

### 11.2 逐组件日志

建议在 `models/archived_frequency/profile_v5_frequency.py` 输出：

| 组件 | 计时与重要性 |
|---|---|
| `GCN base_logits` | 和原 B 的前向时间直接对齐 |
| `A_binary @ x` | 新增第一遍稀疏传播 |
| `feature stack + tiny MLP + gate` | 必须含 CPU/GPU launch 开销 |
| `P[2 scalar]` | 新增第二遍稀疏传播；一次合并两路而不是两次循环 |
| `final logit add + sigmoid` | 排除额外 kernel 过多 |
| `energy archive / decoder / PAVE` | 这几项默认应基本不变 |
| 100步完整 solve | 真实端到端时间；组件值不可机械相加 |

### 11.3 时间预算主报告

固定 T100 只保证**搜索更新数相同**，不保证时间相同。最终必须报告：

- 0.5s/图、1.0s/图的共同时间预算；
- 实际 Mean Size、Mean solve time、P95、最大超时、平均搜索步数、PAVE 后处理耗时；
- 在同样时间下，`B+PAVE`、`B持续搜索+PAVE`、`B持续搜索+rule23`、`v5a_mlp+PAVE`、`v5a_message+PAVE`、`v5a+PAVE`；
- v5 固定 T100 搜索质量若好但耗时大，必须把“质量改进”和“实用增益”分开结论。

**来自 v4 的底线**：F4 的完整 solve 为 B 的 1.88倍，限时平均约59步，说明“多花计算来改 logits”非常容易得不偿失。DAFR 即使降到很小的额外参数量，也不能仅用 Params 判断快慢；稀疏 CPU/GPU 小算子启动成本可能占主导。

---

## 12. 完整实验矩阵、阶段推进与停止条件

| 阶段 | 实验 | 数据 | 训练/测试 | 决策 |
|---|---|---|---|---|
| S0 | `check_adapter.py` 测试算法不变量、零热启动 | 小图 CPU/GPU | 无训练 | 任何不变量失败都先修代码 |
| S1 | 同状态谱 / 分节点概率 / 原生重复 | 训练20图 + 验证100图 | 冻结参数 | 仅研究机制，不能按漂亮频谱选架构 |
| S2 | B / B+FT / MLP / MSG / v5a / 解析ΔE | 训练1000、验证100 | 3 epoch×3种子、T100 | 主质量和信息对照 go/no-go |
| S2b | `v5a_low/high/fixed/noenergy` | 同 S2 | **各独立训练**，先种子0 | 有效后补足3种子，否则不扩张 |
| S3 | on-policy 与共同教师状态混合对照 | 同训练划分 | 3–10 epoch×3种子 | 只在 S2 通过时研究分布偏移 |
| S4 | 完整50 epoch 和公平对照 | 训练1000、验证100 | 3种子、100000更新预算或明确新预算 | 要求测试前冻结超参数与实现 |
| S5 | 官方1000测试图主表 | 测试1000 | 固定T100、0.5s，冻结PAVE | 最终论文证据，严禁在测试图上再挑结构 |
| S6 | 跨图分布/规模、鲁棒性 | 不同尺寸 RB、ER等 | 按事先协议 | 仅在主结论成立后做泛化 |

### 12.1 结构验收阈值（建议预注册）

为了避免“反复调到验证集上 0.01 的正增益”，建议使用明确门槛。以下为**研究项目的建议**，不是统计学上的普适标准：

1. 固定 T100 验证 Mean Size：DAFR 相对 MSG/MLP/B 的平均差为正，至少 2/3 种子同向；任何 0.01 左右的小幅差异若置信区间含零，定性为待确认。
2. 限时 0.5s 验证：必须与 **B持续搜索+冻结PAVE** 比较，且不能靠超过性能较差的 B+FT 偷换基线。
3. 冻结 PAVE 与无 PAVE 的结果均报告；不能将 PAVE 独立产生的 +0.1 左右收益算成频率贡献。
4. 同信息强对照不存在增益时：**No-go 当前频域设计**。允许一次事前指定的信号变化（例如从 x/c 统计改为冲突/阻挡构造），但不允许无限组合试验。
5. 即使频谱理论完全成立，若 Size 与共同时间预算不过关，不升级为论文第二创新点。

### 12.2 如果 S2 又失败，应该如何调整？

按“最小额外假设”排序：

- **先换信号对象**：例如仅对精确边际能量 `ΔE` 做输出端图频投影，或加入已选节点冲突掩码；其他训练协议不变。
- **再换注入条件**：用已知的搜索阶段 `t/T` 做**全图×粒子级标量门控**（同一粒子所有节点共享一个 gate），不在最终投影之后乘节点级 gate；这需要在 `flip_logits`、`sample_states` 和生成训练状态入口增加 `step` 可选参数，作为单独设计版本。
- **最后考虑低成本条件多尺度**：只有单尺度确有收益时才加 `P²`，并通过缓存与单次联合 sparse 乘法控制开销。不要一上来恢复 F4 的复杂双尺度 128通道编码。
- **若依旧没有增益**：可把“频率结构没有提供超出普通邻居统计的任务价值”写成负结果，停止强行包装为创新；把精力转向 PAVE 的独立进一步研究或其他有证据的方向（本报告不修改它）。

---

## 13. 执行级说明：脚本、运行次序、实验产物

本节提供**建议新增的脚本接口**。这些命令当前尚不存在，必须按 §6 实现后才能运行；这是实施规范，不冒称现在可以直接执行。项目服务器以实际本机 `PYTHON` 路径为准。

### 13.1 目录结构

```text
models/
  mis_v5_frequency.py              # 新增；核心微型 output residual
  train_v5_frequency.py            # 新增；teacher states + 日志
  eval_v5_frequency.py             # 新增；复用 v3 frozen-PAVE 协议
  diagnose_v5_frequency.py         # 新增；数学/动作/谱诊断
  profile_v5_frequency.py          # 新增；独占时延分析
train_scripts/legacy/v5_frequency/
  README.md
  smoke.sh
  train.sh
  eval.sh
  eval_budget.sh
  diagnose.sh
  profile.sh
  decide.py
outputs/v5_frequency/
  preflight/
  train_{none,mlp,message,dafr}_seed{0,1,2}/
  eval_fixed_*/
  eval_budget_*/
  diagnostics_*/
  analysis/summary.csv
  analysis/per_seed.csv
  analysis/paired_effects.csv
  analysis/decision.json
```

### 13.2 新 `train.sh`：建议以 v4 脚本为模板

文件 `train_scripts/legacy/v5_frequency/train.sh`：路径变量使用 `ROOT` 与原 `common.sh` 一致。

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=backend:cudaMallocAsync
PYTHON="${PYTHON:-python}"
SEED="${SEED:-0}"
ARM="${1:?use none|v5a_mlp|v5a_message|v5a}"
# B+FT: same seeded v2 none warm-start. v5: add adapter with strict key audit.
WARM="outputs/v2/none_seed${SEED}/checkpoints/best.pt"
mkdir -p outputs/v5_frequency
"$PYTHON" -u models/archived_frequency/train_v5_frequency.py \
  --result_dir "outputs/v5_frequency/train_${ARM}_seed${SEED}" \
  --frequency "${ARM}" --seed "$SEED" --epochs "${EPOCHS:-3}" \
  --warm_start "$WARM" --state_source teacher \
  --graph_batch_size 8 --batch_size 64 --lr 0.0001 \
  --finetune_lr 0.00002 --validation_seed 1017 \
  --train_path models/data/MIS_tensors/RB_small_train --train_samples 1000 \
  --valid_path models/data/MIS_tensors/RB_small_val --valid_samples 100 \
  --test_path models/data/MIS_tensors/RB_small_author_test \
  --num_workers 2 --method RLNN --loss erdoes --problem mis \
  --num_k 20 --num_t 100 --num_tp 50 --num_d 5 \
  --num_l 5 --num_h 128 --mixed_precision no --beta 1.02 --lambd 0.5
```

**实现前须修改**：上面展示共同参数，不包含 v5 的“第1轮冻结骨干”标志；真正的 v5 训练器应在 `ARM != none` 时自动设置 `freeze_backbone_epochs=1`，或在脚本中按 ARM 添加 `--freeze_backbone_epochs 1`。`none` 使用正常的 B+FT 权重；**不能直接把上述脚本当成已完整实现**。建议将输出目录“若存在即报错”以免覆盖既有训练记录。

### 13.3 最小运行顺序

```bash
# 这些都是待新增脚本的期望接口，不是仓库当前已有命令。
export PYTHON=/home/hzeng/envs/rlnn/bin/python

# 1) 算法和数据不变量；任何失败不进入实验
bash train_scripts/legacy/v5_frequency/smoke.sh

# 2) 两卡按 seed/arm 分配，训练状态共享教师
CUDA_VISIBLE_DEVICES=0 SEED=0 bash train_scripts/legacy/v5_frequency/train.sh v5a
CUDA_VISIBLE_DEVICES=1 SEED=0 bash train_scripts/legacy/v5_frequency/train.sh v5a_message
# ... 继续完成 B+FT、v5a_mlp、v5a_message、v5a 的全部种子

# 3) 空闲 GPU 独占：验证FP32无PAVE、FP16无PAVE、FP16冻结PAVE
SPLIT=validation PRECISION=no SAMPLES=100 bash train_scripts/legacy/v5_frequency/eval.sh v5a
SPLIT=validation PRECISION=fp16 SAMPLES=100 bash train_scripts/legacy/v5_frequency/eval.sh v5a

# 4) 同状态／同随机数诊断
bash train_scripts/legacy/v5_frequency/diagnose.sh

# 5) 共同0.5s时间预算与独占时延
bash train_scripts/legacy/v5_frequency/profile.sh

# 6) 计算配对表、置信区间和阶段准入
$PYTHON train_scripts/legacy/v5_frequency/decide.py
```

再次强调：`eval_v5_frequency.py` 应从现有 `models/archived_frequency/eval_v4_frequency.py` 复制控制逻辑。它必须强制 **v3 同种子 PAVE**、H1、候选 cap32、严格改善 cap48、三轮初始交换、四轮修复、八轮平台搜索。原 v4 的 `configure()` 校验不会自动接受 v5 模式，不可简单把旧 `eval.sh` 的 `--frequency` 改名就运行。

### 13.4 每个任务应记录什么

每个 `run.json` 至少包括：`source_sha`、骨干/适配器/PAVE checkpoint SHA256、训练数据签名、验证图 ID、`seed`、搜索随机流、`precision`、`batch`、K/T、`status`、`params`、`train_wall_s`、`mean_size`、`mean_time_s`、`p95_time_s`、`peak_memory_mib`、`feasible_rate`、`mean_search_steps`、`deadline_overruns`。

每图 `per_graph.csv` 至少包括：`graph_id,seed,method,backbone_size,rule_size,final_size,solve_time_s,search_steps,search_time_s,post_time_s,conflicts,selected_nodes`。

每轮 `training.csv` 至少包括：`epoch,step,train_loss,train_energy,train_distance,train_expected_flips,val_size,val_time,alpha_low,alpha_high,gate_low_mean,gate_high_mean,delta_logit_rms,delta_prob_mean_abs,adapter_gradient_norm`。

`analysis` 汇总必须把“同一次 RLNN 搜索、共用起点的后处理比较”和“两个独立求解器产生不同起点的整体比较”分开；不要在叙述中把后者包装成局部后处理的因果收益。

---

## 14. 强制执行的防泄漏、复现及论文呈现规范

1. **官方 1000 测试图在最终结构锁定前不能用来选择 v5 细节**。本次追诊的验证100图之前已参与选权重，是模型选择证据，不是独立测试成绩。
2. **三种子是三次训练/检查点来源**；同图不同随机种子的重复记录不等于新的图样本。报告 `Mean±sample SD`，并给图级配对重采样区间。
3. **如果只看到训练 loss 下降，不能宣称有效**。RLNN 的终极指标是合法 MIS 的 Mean Size 和 Time。
4. **不同模型的 K/T 要相同，PAVE 保持完全一致**；固定时间再单独公平比较，不能只比较参数量或每步耗时。
5. **所有最终输出可行率必须100%**；图冲突边为零。保持原 A.3 decoder 与 beta=1.02，无论频率是否改善都不许私换解码规则。
6. **结构控制要强**：若 `v5a` 只胜 B 而不胜同信息 MSG、廉价 MLP、解析 ΔE，不能说是“频率创新证实”。
7. **零频的图谱分析要区分驻留与非驻留**，不能拿未经去驻留的 97% 和去驻留的 90% 比较成“频率提高90倍”。
8. **收益标签可靠性不足时不得硬上损失**。v4 两半 Spearman≈0.15、R16 仍弱，直接用它指导强监督很可能过拟合噪声。
9. **PAVE 是第一创新点，v5 是第二待验证点**：只冻结使用，不重新训练、不因结果不佳而改变参数或预算，以免无法归因。
10. **实验失败同样是结果**。若 v5 依然无法超过等信息和同时间对照，论文应据实将高低频线标为探索性负结果，不制造理论结论。

---

## 15. 与 2024—2026 文献对齐（正式链接 + GitHub）

下面是 v5 需要直接对照的文献，不重复早期24篇综述中无关的因果/视觉条目。

| 方法 | 正式出处 | 可以参考的机制 | PDF | GitHub |
|---|---|---|---|---|
| **GRACE** | NeurIPS 2025 | 低通/高通专家、条件门控；证明“两路门控”并非新颖贡献 | [PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/eb154abcbfa13d23f01ad3e091e92b08-Paper-Conference.pdf) | [KEAML-JLU/GRACE](https://github.com/KEAML-JLU/GRACE) |
| **WaveGC** | ICML 2025 | 多尺度谱滤波、图小波；在简单版证实前不宜直接导入高复杂度 | [PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/liu25y/liu25y.pdf) | [liun-online/WaveGC](https://github.com/liun-online/WaveGC) |
| **COPT-MT / GCON** | ICML 2026 | 多滤波器图网络已经进入 MIS 等组合优化；必须对标相关先例 | [PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/canturk26a/canturk26a.pdf) | [semihcanturk/COPT-MT](https://github.com/semihcanturk/COPT-MT) |
| **HeroFilter** | NeurIPS 2025 | 不是简单越高频越好；滤波响应要根据任务与图关系选择 | [PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/5d570ed1708bbe19cb60f7a7aff60575-Paper-Conference.pdf) | [zshuai8/HeroFilter](https://github.com/zshuai8/HeroFilter) |
| **PolyGCL** | ICLR 2024 | 可学习谱多项式响应与低高频视图，提醒做容量/信息控制 | [PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/6faf3b8ed0df532c14d0fc009e451b6d-Paper-Conference.pdf) | [ChenJY-Count/PolyGCL](https://github.com/ChenJY-Count/PolyGCL) |
| **RLNN / RLD4CO** | ICML 2025 | 基线目标与多步采样、解码协议 | [PDF](https://proceedings.mlr.press/v267/feng25r/feng25r.pdf) | [Shengyu-Feng/RLD4CO](https://github.com/Shengyu-Feng/RLD4CO) |

相关文献的共同边界：图谱滤波能改变表示，但没有一篇自动证明其在**这个特定 RLNN 动态搜索 + A.3 + 冻结 PAVE**协议下必定提高 MIS Size。v5 的论文价值必须来自严格对照和可信机制，而不是罗列已有频率模块。

---

## 16. 交付给代码执行工具的最终任务说明

> 在源码版本 `eaf290a` 上，先建立 `v5_frequency` 独立目录；实现 §6 的 `V5DAFRGCN` 并只在 `main.py/solvers.py` 添加模式接入；测试所有数学不变量、热启动一致性、图置换等变性、轨迹独立性和块对角梯度；在所有测试通过后，按 §9 的 B / B+FT / MLP / MSG / DAFR / 解析能量对照，以共享冻结教师状态和 3 种子 6000 更新进行验证；先报告 FP32 固定 T100 的无 PAVE 骨干指标，再报告 FP16 与 0.5 秒冻结 PAVE 指标；独占GPU测时，不超过预算不合格时停止扩大；所有 v5 结果进入新的 JSON/CSV，不覆盖 v3、v4 权重；生成分析表与阶段结论。当前全部 v5 数值留空，不预填预期收益。

**本轮完成判据**：得到一份能回答“是否有额外信息价值？是否优于普通消息？是否能在共同时间预算获益？为何有效或为何失败？”的可复算诊断与对照报告，而不是仅有一条 Size 数字。

---

**简短结论**：最新实验支持放弃“128通道、输入端多尺度双支路堆叠”这一条当前实现路线，但**没有**证明图频率这一研究方向必然无效。v5 的合理赌注是低成本的输出端决策频带约束，并且必须先击败同信息的普通消息和廉价状态控制。若做不到，就果断 No-go，而不是再用更复杂的网络掩盖机制证据缺口。
