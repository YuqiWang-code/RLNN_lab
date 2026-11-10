# RLNN_lab 图频域模块专项调研与 v4 重设计方案

**研究范围：2024—2026 年相关论文 + RLNN_lab 最新代码、实验日志、消融；保留 PAVE-Stats H1，不再研究 Archive。**  
**编写日期：2026-11-09**  
**代码基准：** [`YuqiWang-code/RLNN_lab`](https://github.com/YuqiWang-code/RLNN_lab)，`main` 提交 [`63a353e282fbc`](https://github.com/YuqiWang-code/RLNN_lab/commit/63a353e282fcbcda58b46302fa1ff8f85e802bd3)（最后提交时间 2026-11-08 22:57 UTC）。  
**文档性质：** 技术调研 + 结构重设计 + 可执行实验计划；不是已完成的 v4 实验报告。

> **总判断：** 现有“高低频分解→两路投影→门控→GCN”尚未给 RLNN 提供独立的有效搜索能力。当前最好的成果是 **v3：微调 RLNN + 合法中性平台搜索 + PAVE-Stats H1**。建议**冻结 PAVE 算法、权重、候选集合规则及后处理配置**，仅围绕 **RLNN 的搜索状态编码**重新研究频域模块。下一版不优先追求更复杂的滤波器，而是检验：**哪些图频率成分能够预测 MIS 的冲突消除、局部可扩展性与未来交换机会，并因此改善翻转决策？**

---

## 0. 一页读懂结论

| 问题 | 从已提交的实验得出的结论 | 下一版处理 |
|---|---|---|
| 高低频有没有实现？ | v1、v2 已实现；v3 的 PAVE 评分器也比较了普通消息与频率编码 | 不再重复“做没做过”的实验 |
| 为什么成绩没提升？ | 一阶低高频和已有邻居统计可线性互相恢复；多尺度与交换图版本计算开销增加，但未产生更好 Size | 从信息贡献、搜索收益和时间成本三个方面验证 |
| v2 SEGF 完全无用吗？ | 不能下此结论；某些单种子消融达到相近/略高 Size，但完整 SEGF 的三种子结果更差 | 保留合理部件，重构输入、注入和门控 |
| 频率是否越高越好？ | **不是**。频率衡量图上差异，不等于 MIS 解质量；简单低频、高频均不保证有效翻转 | 引入与合法性、释放机会相关的状态描述和任务诊断 |
| PAVE 怎么处理？ | v3 PAVE-Stats H1 在同一起点下有可重复的训练增益 | **不改 PAVE**；仅作为固定后处理与最终评估 |
| 要先做什么？ | 当前无法证实“多频分支比同信息普通消息更好” | 先做频域信息检验和零扰动 adapter，再做任务条件多尺度 |

### 最终推荐

- **研究对象**：搜索阶段的 **Decision-Conditioned Graph Frequency Adapter（DCGFA，暂定名）**；不是修改 PAVE 的频率头。
- **一条主线，两层递进**：`V4-A 低风险适配器` → `V4-B 交换机会条件频域适配器`。只有 V4-A 消融证明分频确实有价值，才推进 V4-B。
- **最重要的对照**：`相同结构统计 + 相同参数预算的普通 MLP/消息分支`，以及 `完全相同时间预算下允许继续搜索的 RLNN+PAVE`。
- **决策阈值**：若连续三个训练种子仍未超过参数匹配的非频域对照，或只在大幅加时后提升，则不继续将其包装为独立创新点。

---

## 1. 项目与指标：哪些结果才是可比证据？

### 1.1 任务、张量与原目标

RLNN 解决最大独立集（MIS）：给定无向图 `G=(V,E)`，选择尽可能多的节点，保证选中节点之间没有边。每个候选状态 `x∈{0,1}^N`；同时并行 `K` 条搜索轨迹，实际网络输入 `[N,K,1]`。**这里不是图像 `H×W×C` 特征图，不能沿 `N×K` 做二维 FFT/Haar。**

基线能量函数：

\[
H(x)=-\sum_i x_i+\beta\sum_{\{i,j\}\in E}x_ix_j,\quad \beta=1.02.
\]

GCN 输出节点的**翻转概率**（不是直接选中概率），推理时 `K=20`、`T=100`。训练采用一步期望能量与期望翻转距离正则。历史最低能量状态经原 A.3 梯度翻转解码，得到合法独立集。

**基线策略不允许被频域实验悄悄改变**：二值状态、Bernoulli 翻转、距离正则、搜索轨迹数、步数、历史保存、A.3 解码都保留。相关源码：[`models/models.py`](https://github.com/YuqiWang-code/RLNN_lab/blob/63a353e282fcbcda58b46302fa1ff8f85e802bd3/models/models.py)、[`models/solvers.py`](https://github.com/YuqiWang-code/RLNN_lab/blob/63a353e282fcbcda58b46302fa1ff8f85e802bd3/models/solvers.py)。

### 1.2 已核实的主要定量结果（不同版本不能混成同一训练协议）

**v1：50 epoch、三种子训练，完整 RLNN 分频版本**

| 版本 | Mean Size ↑ | 平均求解时间（秒/图） ↓ | 结论 |
|---|---:|---:|---|
| v1 RLNN | 19.515 | 0.307 | v1 基线 |
| v1 Frequency Dynamic | **19.406** | **0.467** | 相对基线 −0.109；时间增加 |
| v1 Stats（只加 MIS 统计） | 19.448 | 0.418 | 统计输入本身也没证明优势 |

**v2：从原 RLNN 同种子检查点启动、仅 3 epoch 的适配器训练；不是完整从头训练**

| 版本 | Seed 0 | Seed 1 | Seed 2 | 三种子平均 Size ↑ | 运行时间参考 |
|---|---:|---:|---:|---:|---|
| RLNN（v2 `none`） | 19.533 | 19.546 | 19.558 | **19.546** | 约 0.235 s/图 |
| v2 完整 SEGF | 19.513 | 19.496 | 19.518 | **19.509** | 约 0.513 s/图 |
| 完整 SEGF − RLNN | −0.020 | −0.050 | −0.040 | **−0.037** | 约 2.2 倍 |

> v2 是“短期适配 / 微调”比较：第 1 轮只训练 adapter，第 2—3 轮解冻；与 50 epoch 从头训练不同。**不能凭这三轮认定所有频域方法都失败**。但完整 SEGF 在三个种子均下降，而且耗时明显增加，足以说明**该具体设计目前不划算**。来源：[v2 协议](https://github.com/YuqiWang-code/RLNN_lab/blob/63a353e282fcbcda58b46302fa1ff8f85e802bd3/train_scripts/legacy/v2/README.md)、[`outputs/v2/`](https://github.com/YuqiWang-code/RLNN_lab/tree/63a353e282fcbcda58b46302fa1ff8f85e802bd3/outputs/v2)。

v2 部分仅 seed 0 的消融有：`segf_static1=19.550`、`segf_static2=19.522`、`segf_exchange=19.539`、`segf_high=19.538`、`segf_fixed=19.546`（同 seed 基线为 19.533）。**某一组偶然超过基线 0.017 不构成稳定改进**，也不能与三种子平均混算。

**v3：PAVE 已固定并取得正收益，但“频率版评分器”仍不优于统计版**

| v3 固定搜索量、相同 RLNN 初始解 | 3 seed Mean Size ↑ | 时间（秒/图） ↓ | 说明 |
|---|---:|---:|---|
| 微调 RLNN、无后处理 | 19.54667 ± 0.01210 | **0.25581** | 固定起点 |
| + rule23 | 19.60300 ± 0.00964 | 0.25798 | 规则局部改善 |
| + 随机平台搜索 | 19.62433 ± 0.01415 | 0.30767 | 中性动作本身有收益 |
| + PAVE-Stats H1 | **19.66333 ± 0.01185** | **0.33663** | 当前主方法 |
| + PAVE-Message H1 | 19.65633 ± 0.00945 | 0.36880 | 额外节点消息较慢 |
| + PAVE-Frequency H1 | 19.65233 ± 0.00709 | 0.37061 | 对 Stats −0.011，对 Message −0.004 |

`PAVE-Message` 与 `PAVE-Frequency` 共享 8 维节点特征、候选池化、参数量（20,737 个评分器参数）、训练数据和标签。前者输入 `[F, MF]`，后者输入 `[(F+MF)/2, (F−MF)/2]`，信息线性等价。这个**控制实验没有支持频率参数化本身**。统计头仅 3,713 个参数，且更快。来源：[v3 正式分析](https://github.com/YuqiWang-code/RLNN_lab/blob/63a353e282fcbcda58b46302fa1ff8f85e802bd3/docs/temporary/RLNN_v3_Results_Analysis.md)、[汇总](https://github.com/YuqiWang-code/RLNN_lab/blob/63a353e282fcbcda58b46302fa1ff8f85e802bd3/outputs/v3/summary.txt)。

**v3 统一 0.5 秒预算**：PAVE-Stats H1 为 **19.64867 ± 0.01159、0.38512 秒/图**；“微调 RLNN 持续搜索 + rule23”为 **19.63500 ± 0.07448、0.44932 秒/图**。这是三种子均值的取舍，PAVE 并非对强规则的每个种子都占优。**PAVE 不改，也不把 PAVE 的收益误算到频域模块上。**

### 1.3 正确的实验目标

- **直接目标**：同搜索量 `K=20,T=100` 下，频域 RLNN 输出更大的合法独立集；同时记录额外耗时。
- **组合目标**：给同一个**冻结的 v3 PAVE-Stats H1 后处理**，使新 RLNN+PAVE 超过微调 RLNN+PAVE。
- **效率目标**：给定 0.5 秒/图的**实际总时间**，相比延长搜索的 RLNN 和 `RLNN+rule23` 仍有优势。
- **诊断目标**：证明分频与交换机会有关，且**分支的训练贡献**超过仅添加结构统计、参数量或随机扰动。

---

## 2. 2024—2026 年相关论文：PDF、GitHub 与值得借鉴的部分

### 2.1 筛选原则与阅读优先级

- 选择正式主会 / 已正式发表的相关工作，优先 ICML、NeurIPS、ICLR、AAAI、CVPR、TPAMI；GCON（Learning on Graphs）是**直接任务相关**的额外必读，不误写为 CCF A。
- 提供**可直接下载的论文 PDF**；GitHub 是检索到的作者公开仓库。**“公开源码”不代表本项目已经复现，部分代码与当前 torch/PyG 版本不兼容。**
- 不把视觉上的高低频含义直接搬到 MIS；视觉论文主要借鉴**分支职责、动态门控、跨频计算与融合成本控制**。

### 2.2 核心论文总表（16 篇）

| 编号 | 论文（正式年份 / 来源） | 与本项目最相关的思想 | PDF 下载 | 作者 GitHub |
|---|---|---|---|---|
| G01 | **WaveNet: Tackling Non-stationary Graph Signals via Graph Spectral Wavelets**（AAAI 2024） | 图小波、多分辨率、非平稳信号 | [PDF](https://ojs.aaai.org/index.php/AAAI/article/download/28781/29496) | [WaveNet](https://github.com/Bufordyang/WaveNet) |
| G02 | **PolyGCL: Graph Contrastive Learning via Learnable Spectral Polynomial Filters**（ICLR 2024） | 可学习低/高通；图谱视图 | [PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/6faf3b8ed0df532c14d0fc009e451b6d-Paper-Conference.pdf) | [PolyGCL](https://github.com/ChenJY-Count/PolyGCL) |
| G03 | **SLOG: An Inductive Spectral Graph Neural Network Beyond Polynomial Filter**（ICML 2024） | 自适应谱滤波、归纳泛化、避免仅固定多项式 | [PDF](https://raw.githubusercontent.com/mlresearch/v235/main/assets/xu24aa/xu24aa.pdf) | [SLOG](https://github.com/Hsu1023/SLOG) |
| G04 | **Spatio-Spectral Graph Neural Networks**（NeurIPS 2024） | 空间与频域互补，关注远距离信息 | [PDF](https://arxiv.org/pdf/2405.19121) | [s2gnn](https://github.com/sigeisler/s2gnn) |
| G05 | **A General Graph Spectral Wavelet Convolution via Chebyshev Order Decomposition（WaveGC）**（ICML 2025） | scaling 与 wavelet、多尺度、奇偶 Chebyshev | [PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/liu25y/liu25y.pdf) | [WaveGC](https://github.com/liun-online/WaveGC) |
| G06 | **Graph Neural Networks Are More Than Filters: Revisiting and Benchmarking from A Spectral Perspective**（ICLR 2025） | **频域行为必须单独测量**，不能从网络结构推断 | [PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/a36c3dbe676fa8445715a31a90c66ab3-Paper-Conference.pdf) | [Spectral-benchmark](https://github.com/yushundong/Spectral-benchmark) |
| G07 | **GRACE: Graph Few-Shot Learning via Adaptive Spectrum Experts and Cross-Set Distribution Calibration**（NeurIPS 2025） | 低、高通专家 + 节点条件门控；**近似结构先例** | [PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/eb154abcbfa13d23f01ad3e091e92b08-Paper-Conference.pdf) | [GRACE](https://github.com/KEAML-JLU/GRACE) |
| G08 | **HeroFilter: Adaptive Spectral Graph Filter for Varying Heterophilic Relations**（NeurIPS 2025） | 局部异构条件下的谱适配，不预设固定高/低频偏好 | [PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/5d570ed1708bbe19cb60f7a7aff60575-Paper-Conference.pdf) | [HeroFilter](https://github.com/zshuai8/HeroFilter) |
| G09 | **MORGAN: To Bridge Mixture of Experts and Spectral Graph Neural Network**（AAAI 2026） | 频段专家 + 门控；注意全图特征分解成本 | [PDF](https://ojs.aaai.org/index.php/AAAI/article/download/39553/43514) | [Morgan](https://github.com/lihuiliullh/Morgan) |
| G10 | **Can Computational Reducibility Lead to Transferable Models for Graph Combinatorial Optimization?**（ICML 2026） | **MIS 上已有 GCON 多滤波器结构**，直接竞争先例 | [PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/canturk26a/canturk26a.pdf) | [COPT-MT](https://github.com/semihcanturk/COPT-MT) |
| V01 | **FreqFusion: Frequency-aware Feature Fusion for Dense Image Prediction**（TPAMI 2024） | 高/低频各司其职、频率选择和残差融合 | [PDF](https://arxiv.org/pdf/2408.12879) | [FreqFusion](https://github.com/Linwei-Chen/FreqFusion) |
| V02 | **Frequency-Adaptive Dilated Convolution for Semantic Segmentation（FADC）**（CVPR 2024） | 按局部状态选频带与感受野，不直接照搬二维膨胀卷积 | [PDF](https://arxiv.org/pdf/2403.05369) | [FADC](https://github.com/Linwei-Chen/FADC) |
| V03 | **Frequency Dynamic Convolution for Dense Image Prediction（FDConv）**（CVPR 2025） | 按内容调制频率响应；参数与频带多样性控制 | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Frequency_Dynamic_Convolution_for_Dense_Image_Prediction_CVPR_2025_paper.pdf) | [FDConv](https://github.com/Linwei-Chen/FDConv) |
| V04 | **F2Net: A Frequency-Fused Network for Ultra-High Resolution Remote Sensing Segmentation**（CVPR 2026） | 异构分支、跨频融合、训练梯度平衡诊断 | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_F2Net_A_Frequency-Fused_Network_for_Ultra-High_Resolution_Remote_Sensing_Segmentation_CVPR_2026_paper.pdf) | [F2Net](https://github.com/easm002/F2Net) |
| C01 | **Towards a General Recipe for Combinatorial Optimization With Multi-Filter GNNs（GCON）**（LoG 会议、PMLR **2025 出版**） | 图滤波器组、局部注意力与组合优化；**LoG 非 CCF A** | [PDF](https://raw.githubusercontent.com/mlresearch/v269/main/assets/wenkel25a/wenkel25a.pdf) | [copt](https://github.com/WenkelF/copt) |
| C02 | **Regularized Langevin Dynamics for Combinatorial Optimization（RLNN）**（ICML 2025） | 原基线；一步能量、翻转正则、五层 GCN | [PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/feng25r/feng25r.pdf) | [RLD4CO](https://github.com/Shengyu-Feng/RLD4CO) |

> **代码状态说明**：上述 GitHub 均有对应公开仓库或作者项目，未逐个执行训练；LOHA 等未核验作者源码的工作未强行写成“已开源”。GCON 的会议是 Learning on Graphs；ICML 2026 的 COPT-MT 才属于正式 ICML 论文。这里总表按正式发表年份而不是 arXiv 首次出现年份归档。

### 2.3 最值得深入的 8 篇：它们到底告诉我们什么？

**A｜G07 GRACE（NeurIPS 2025）**  
在图节点分类中用低通专家、高通差异专家和节点条件门控。**结论不是“照搬 GRACE”，而是承认已有这套结构，下一版必须提出 MIS 特有的动态候选状态/局部交换收益证据。** 它的交叉集合校准不对应当前 MIS，没有必要搬来。

**B｜G05 WaveGC（ICML 2025）**  
支持多尺度滤波与不同尺度特征的联合表达。我们可以借鉴“传播尺度差分”，但第一阶段只选一阶、二阶，先避免多项式阶数和显存急剧增加。原 WaveGC 的 Chebyshev 波段可作为后期对照，而不是第一版必备依赖。

**C｜G08 HeroFilter（NeurIPS 2025）**  
提醒“高频与异配关系并不是单一、固定、单调对应”。在 MIS，1–1 冲突边、1–0 合法边、0–0 空置边都有不同作用；不应该预设“高频一定重要、低频一定没有用”。门控至少需要在状态与约束条件上产生可测变化。

**D｜G06 Spectral-benchmark（ICLR 2025）**  
网络的非线性本身可以改变频率分布，即使输入经过滤波，也不能据此声称输出保持该频带。我们必须记录**实际分支输出、融合后和最终 logits**的谱统计，并通过分支消融证实作用。

**E｜G10 COPT-MT（ICML 2026）和 C01 GCON（LoG/PMLR 2025）**  
这是最需要面对的**新颖性边界**：已有多滤波器图网络参与 MIS 求解。它们与 RLNN 的区别在于 GCON 类方法常从图结构编码预测选中分数，而 RLNN 是 **当前二值状态 → 迭代翻转概率**。我们的可研究差异应是**动态搜索态下的状态条件频域决策**，不能写“首次将图小波引入 MIS”。

**F｜G09 MORGAN（AAAI 2026）**  
谱频段可以用专家机制分工，但直接做全图特征分解、频率聚类和多专家，会放大每步搜索成本。在 20 条轨迹×100 步的 RLNN 中，应先采用 `P` 与 `P²` 的稀疏局部实现，除非验证显示局部近似不足。

**G｜V02 FADC（CVPR 2024）和 V01 FreqFusion（TPAMI 2024）**  
可以迁移**局部频率选择、不同分支适合不同的计算路径**，但不是将图节点排成图像去做二维 FFT；图结构决定邻域。图上的高频不等于边缘目标或视觉纹理。

**H｜V04 F2Net（CVPR 2026）**  
最适合借鉴的是**分支梯度强度、融合权重及有效特征利用的监控**。不要直接将图像语义对齐 loss 用到无标签 MIS；如果加辅助损失，需要定义它和翻转/解码收益的明确联系。

### 2.4 对创新边界的结论

已经有明确先例：`低频/高频分开处理`、`多尺度图滤波`、`谱专家门控`、`动态图任务中可学习滤波`、`多滤波器用于组合优化`。**这些现成构件不能单独称为新贡献。** 候选贡献需要是“**动态 MIS 搜索态 + 特定图约束机会信号 + 受控的频域适配机制 + 同预算解质量证据**”。在实验前只能作为研究假设。

---
## 3. 逐段审查当前高低频代码：真正的薄弱点在哪里？

阅读范围包括：[v1 `mis_v1.py`](https://github.com/YuqiWang-code/RLNN_lab/blob/63a353e282fcbcda58b46302fa1ff8f85e802bd3/models/mis_v1.py)、[v2 `mis_v2_frequency.py`](https://github.com/YuqiWang-code/RLNN_lab/blob/63a353e282fcbcda58b46302fa1ff8f85e802bd3/models/archived_frequency/mis_v2_frequency.py)、[v2 频域诊断](https://github.com/YuqiWang-code/RLNN_lab/blob/63a353e282fcbcda58b46302fa1ff8f85e802bd3/models/archived_frequency/diagnose_v2_frequency.py)、[v3 `mis_v3_value.py`](https://github.com/YuqiWang-code/RLNN_lab/blob/63a353e282fcbcda58b46302fa1ff8f85e802bd3/models/mis_v3_value.py) 及 [v3 模型选择](https://github.com/YuqiWang-code/RLNN_lab/blob/63a353e282fcbcda58b46302fa1ff8f85e802bd3/outputs/v3/summary.txt)。

### 3.1 v1：相同邻域信息重复编码，融合难以体现新增能力

v1 `state_features()` 的统计输入为：

1. 当前节点是否选中 `x`；
2. `log(1+degree)`；
3. 当前已选邻居比例 `m=D^{-1}Ax`；
4. 归一化的一步能量改变量 `gain`。

随后构造：

\[
x_L=(x+m)/2,\qquad x_H=x-x_L=(x-m)/2.
\]

因为 `x=x_L+x_H`、`m=x_L-x_H`，**只用一阶滤波时 `[x_L,x_H]` 与已有 `[x,m]` 线性等价**。因此分频只是已知信号的重新参数化；训练也许能因此获益，但必须与 `Stats MLP / 普通消息` 做参数匹配对照。

v1 使用的门控形式为：

\[
h=h_{raw}+g\,\operatorname{SiLU}(W_Lx_L)+(1-g)\,\operatorname{SiLU}(W_Hx_H).
\]

**可直接看到的问题**：
- 两路**互补竞争**：当 `g` 增大时，高频权重必减小；然而局部冲突和可扩展性可能同时需要两路信息。
- 没有显式小残差初始化、分支尺度校准和对主干输出的保护；不同种子会从不同的表示几何进入原 GCN。
- 一阶分频没有带来明确的**新的邻域阶数或动作机会信息**，无法解释为什么应改善 MIS。

以上是**结构问题**；不能在未做消融前断言它们分别导致了多少损失。

### 3.2 v2 SEGF：更复杂了，但问题仍没有解决

`models/archived_frequency/mis_v2_frequency.py` 实现了：

- 原始 RLNN `Linear(1,128)` 保留，并以小残差系数添加适配器；
- 四个节点信号：`x`、截断到 2 的已选邻居数、`log_degree`、`log1p(released)`；
- 原图上的一阶、二阶 lazy-walk；
- `features−P(features)` 和 `P(features)−P²(features)` 两个差分；
- `exchange_band()`：利用当前状态的选中、唯一阻挡节点标志构造**条件邻域差分**；
- 两路小 MLP + LayerNorm、独立 sigmoid 门控 `g_low,g_high`，以及可学习残差系数（初值 0.01）。

这些改进说明你**已经尝试过“多尺度、交换条件、自适应门控”**，下一版不能再把它们写成一个从未尝试过的新主意。真正应该追问“做了之后为什么仍不行”。

#### v2 薄弱点 A：`exchange_band` 不是可以直接解释的标准图谱频带

代码中的 `exchange_band()` 先按节点状态构造掩码，再沿原图边传播，最后以 `released + release` 作局部分母，求差。这个运算**有状态条件、带方向性的掩码与局部归一化**，不能直接拿 `I−P` 的拉普拉斯特征值理论，把它说成固定谱域的“高频子带”。

更合适的术语：**交换条件邻域差异（exchange-conditioned discrepancy）**。如果要称其为**动态图谱滤波**，必须先给出明确的**对称、非负的图权矩阵**及归一化拉普拉斯，并核对谱响应；否则它是任务相关图消息算子而非标准谱滤波器。

**改进要求**：将“严格的静态图多尺度分频”和“状态相关的交换机会描述”分离，再研究后者如何条件调制前者。这样每个部件的数学含义与计算贡献才清楚。

#### v2 薄弱点 B：现有 `released` 主要是候选生成线索，不是候选收益

设当前状态 `x`，已选邻居数 `c_i=∑_{j∈N(i)}x_j`。v2 的 `release_j=1` 代表未选中且 `c_j=1`；`released_i` 统计选中节点 i 邻域内这类节点的数量。

这个统计容易识别**“移除 i 可能释放多少邻居”**；但不能判断这些邻居是否互相冲突，也不能判断后续合法独立集最终会增加多少。因此：

- `released_i` 大，未必有合法的 `1→2` 交换；
- 合法的 `1→1` 中性动作也可能打开后续两步的增益，这正是 v3 PAVE 的主要发现；
- 训练样本中的 `x` 可能不合法，不能直接将 `release` 视为实际可执行的合法交换。

**改进要求**：把“释放候选数”“释放候选之间是否兼容”“中性动作之后能否继续改善”分成不同诊断，不拿一个 `released` 值代替全部交换潜力。

#### v2 薄弱点 C：交换频带的 AUC 高，并不意味着求解质量高

`outputs/v2/diagnostics_frequency_segf/run.json`：

| 时刻 | Exchange discrepancy 的 AUC（约） | 动态 gate_low | 动态 gate_high |
|---|---:|---:|---:|
| t=25 | 0.965 | 0.862 | 0.805 |
| t=50 | 0.970 | 0.861 | 0.805 |
| t=75 | 0.965 | 0.861 | 0.805 |
| t=100 | 0.961 | 0.861 | 0.805 |

AUC 的标签是 `diagnose_v2_frequency.py` 中由**交换候选规则生成的节点掩码**，它衡量**能否识别被规则标为候选的节点**；不是能否找出未来收益更高的动作。高 AUC 可能部分来源于同一个“唯一阻挡/释放”定义，同时用于构造特征和标签。

因此这项诊断应写为：**`exchange-candidate node AUC ≈ 0.96`，不写成 `gain prediction AUC ≈ 0.96`**。

门控在 25–100 步之间变化很小，提示**当前门控可能缺乏阶段适应性**；但是只看平均值无法判断节点级差异，仍需按冲突数、阶段、翻转类型分层看门控分布。

同一诊断中，残差相对初始投影范数大约是低频 4%、高频 5%。它并非完全不工作；但也不足以推出下游的最终质量有改善。

#### v2 薄弱点 D：训练目标仍是一跳期望能量

与 v1 相同，v2 继续使用 `erdoes`：

\[
\mathcal L_{RLD}
=\mathbb E_{q_\theta(x'\mid x)}H(x')
+\lambda(\sum_i p_i-d)^2.
\]

它训练的是**下一步平均能量**，没有直接要求“某个频段让后续合法解更大”，也没有度量多步中性平台搜索。模型可能学会降低训练损失，却未学到有效的交换机会。

这不是 RLD 论文的错误，而是**频域模块的贡献尚未和新目标对齐**。后续如加辅助目标，必须明确其训练对象来自训练图，且不会改动原 Bernoulli 条件独立性。

### 3.3 v3：最有价值的控制实验是“信息等价、结构不同”

v3 同一批节点信号 F：

```text
普通消息版：  [F, M F]
频域版：    [(F+MF)/2, (F-MF)/2]
```

二者可互相线性恢复；其区别主要是**分支参数化/非线性处理方式**。实验已在同起点、相同训练数据和参数量下完成，结果频率版 `19.65233` 低于普通消息版 `19.65633`，且二者都低于 Stats 版 `19.66333`。

这是一个比“Frequency vs 原 RLNN”更强的结果：**高低频换坐标本身，没有形成独立收益**。因此 v4 要么设计可诊断的额外状态信息和真正有效的条件处理；要么承认该方向不适合当前资源与任务。

### 3.4 具体薄弱点按优先级汇总

| 级别 | 现象或局限 | 支持证据 | v4 对策 |
|---|---|---|---|
| P0 | 一阶分频与已有局部统计等价 | v1 公式；v3 `[F,MF]↔[L,H]` 控制 | 参数匹配、同信息对照；不要只靠换坐标 |
| P0 | 在训练目标下收益未体现 | v1/v2 三种子 Size 下降 | 把训练/诊断同未来解质量区分 |
| P0 | 频域/交换候选识别与收益错配 | v2 AUC 标签定义；v3 中性动作有效 | 加入“候选收益”或“平台可扩展性”验证 |
| P0 | 计算开销显著 | v2 SEGF 时间约为原模型 2.2 倍 | 在低通维度、传播次数、更新频率上控制成本 |
| P1 | 混淆严格谱滤波与动态图差异 | `exchange_band()` 归一化与状态掩码 | 两路数学定义分离，不能给任意差分贴谱频率标签 |
| P1 | 门控阶段变化不明显 | v2 gate 均值 t25–100 几乎不变 | 节点 × 阶段 × 局部状态分层诊断和独立门控消融 |
| P1 | 两路信息冗余/受主干残差吸收 | v1/v3 及 ICLR 2025 频域基准 | 监测分支范数、logit 贡献和置零干预 |
| P1 | v2 短期微调不足以排除其他训练策略 | v2 只 3 epoch | 稳定性预检后才做严格等预算 50 epoch |
| P1 | 搜索中大量状态不合法 | RLNN 允许临时冲突 | 合法性条件与交换机会描述分开，不把代理量当真实动作 |
| P2 | 仅用一张图规模训练、测试 | 当前主要 RB200–300 | 通过后再做 RB-large/ER 的泛化验证 |
| P2 | 数值和 CUDA 耗时易波动 | 分布式/不同卡负载及缓存 | 固定流、同步计时、同设备空闲重复测试 |

---

## 4. 设计原则：先解决什么，不解决什么？

### 4.1 必须保留的部分

1. **冻结 PAVE-Stats H1**：其评分器权重、输入 24 维统计、合法中性平台搜索 H1、外层最多 8 轮、候选上限与选择协议不更改。这里只讨论 RLNN 的网络表达。
2. **保留 RLNN 原输出契约**：当前图 `G`、候选状态 `X:[N,K]` → 节点翻转 `logits:[N,K]`，外部仍通过 sigmoid+Bernoulli 搜索。
3. **保留原 RLD 训练目标作为主 loss**，避免一开始把新辅助目标、模型结构和训练策略全变了。
4. **保留原五层残差 GCN 与 checkpoint 参数键**，新增模块采用残差式插入，便于继承已有权重。
5. **先验证表征收益，再考虑端到端联合训练**。PAVE 本身不参与修改和反向传播。

### 4.2 当前不建议再做的方案

- 直接在 `[N,K]` 上做二维 FFT / Haar；K 是互相独立的轨迹，不是空间宽度。
- 单纯堆 4、8、16 个频带，声称频带更多一定更强。
- 把图像 `LH/HL/HH` 的方向概念无修改地用于一般无向图。
- 使用原始 `exchange_band` 就宣称得到了可解释的图拉普拉斯高频。
- 把高频能量最大化、低频能量最小化当成 MIS 监督。
- 把 v3 PAVE 的损失/结构一并改掉，从而让两个创新点无法独立消融。
- 在固定 K/T 比较表里省略分频、缓存和 PAVE 后处理的额外耗时。

### 4.3 需要验证的核心假设

**H1（信息假设）**：对动态二值状态，简单的一跳平均与差分没有足够的新增信息；跨一跳与二跳的**局部占用、冲突和释放机会的尺度关系**可能更有判别力。

**H2（机制假设）**：当节点已选/未选、存在冲突/不存在冲突、处于搜索早期/晚期时，需要的频率处理方式不同；节点状态条件门控可能比固定融合有效。

**H3（目标假设）**：只有当频域增量改变了有益翻转的排序、冲突修复速度或后续可行解的可扩展性，才会改善最终 Mean Size；图频率 AUC 或更大分支输出都不是充分条件。

**H4（效率假设）**：如果额外的图传播不够节省，收益即使略为正，也可能在 0.5 秒预算内被消耗掉。

这些都是**可证伪假设**，不是已被论文或当前日志证明的结论。

---
## 5. 主推结构：DCGFA——决策条件图频域适配器（v4-A）

> **定位**：在 RLNN **每次生成翻转 logits 之前**提供小幅、可控的图频域增量。PAVE 的结构、权重、后处理流程完全不动。
>
> **先实现低风险版 v4-A；不保证提高指标。** 如果可验证地超过同信息、同参数对照，才升级 v4-B。

### 5.1 不再仅分解二值 `x`，而是分解有任务含义的动态图信号

给定当前状态 `x_i∈{0,1}`，原图度数 `d_i`，被选邻居数 `c_i=∑_{j∈N(i)} x_j`，定义三个核心动态图信号：

1. **状态通道** `z_i^{(x)}=x_i`，区分当前已选与未选；
2. **约束通道** `z_i^{(c)}=log(1+c_i)`（或使用 `c_i/max(d_i,1)` 做对照），表达局部冲突/阻挡强度；
3. **释放通道** `z_i^{(r)}=log(1+r_i)`，其中

   \[
   r_i=x_i\sum_{j\in N(i)}\mathbf1[x_j=0\;\land\;c_j=1].
   \]

   它是 v2 已使用的“唯一阻挡节点释放数”代理量；**不是新发明的统计，也不等于真实合法交换数**。

另保留静态 `log(1+d_i)` 和精确局部能量翻转差：

\[
\Delta H_i=(1-2x_i)(\beta c_i-1).
\]

**这几个量的分工**：

| 信号 | 对应的数学信息 | 建议输入位置 |
|---|---|---|
| `x` | 当前节点选中状态 | 保留原 GCN 输入；同时分频 |
| `c` | 已选邻居数、局部冲突或加入阻挡 | 分频 + 门控 |
| `r` | 节点被移除后可能释放的一跳节点数 | **先作为门控对照；通过后再分频** |
| `log degree` | 静态邻域规模 | 门控，不强制分频 |
| `ΔH_i` | 单节点翻转的当前能量变化 | 门控与诊断，不等于最终解码收益 |

**第一版建议只对 `[x, log1p(c)]` 两个通道做多尺度滤波**，把 `r` 加到门控。原因是 v2 已经计算过 `r`，且完整模型不成功；先降低信号冗余、传播成本和梯度干扰。只有确实看到释放信息的贡献，再使 `r` 进入频域主分支。

> **边界**：RLNN 的 `x` 在搜索时经常不合法。上面的 `c` 和 `r` 都是**原始状态的局部描述**，不能当作当前状态已经可以执行的合法 1→2 或 1→1 动作。对合法解后的真实交换，应由冻结的 PAVE / 规则检查完成。

### 5.2 一阶、二阶的严格图滤波定义

令原图的二值无向邻接矩阵为 `A`（**注意：项目 MIS 能量用双向边权 `1/2`，频域邻接应还原为二值图**）。`D_ii=∑_j A_ij`。定义惰性随机游走算子：

\[
P=\frac12(I+D^{-1}A).
\]

孤立节点定义 `D^{-1}A` 在该点作用为恒等，因而 `Pz=z`，防止除零。

给定节点信号 `Z∈R^{N×K×C}`，计算：

\[
Z_1=PZ,\quad Z_2=P^2Z,
\]

\[
\boxed{L_2=Z_2},\qquad
\boxed{B_1=Z_1-Z_2},\qquad
\boxed{B_0=Z-Z_1}.
\]

满足严格重构：

\[
\boxed{Z=L_2+B_1+B_0}.
\]

- `L₂`：两步平滑后的**低频来源**；
- `B₁`：两个传播尺度之间的**中间尺度残差**；
- `B₀`：原始状态相对一步邻域的**高频来源**。

模型只做两条可学习主分支：

- **Low branch** 编码 `L₂`（较平滑的局部状态）；
- **High branch** 同时编码 `[B₀,B₁]`，保留“近邻差异”与“一步到二步差异”。

这里是**非正交、软分带**，不要写“严格分成互不相关的 3 个正交频段”。如果输入中有节点度加权与无自环定义，需保持全文公式和实际代码一致。

对图拉普拉斯的谨慎解释：在非孤立部分，`P` 与 `I−L_sym/2` 相似，其谱响应是 `1−λ/2`。这使一阶滤波具有明确的低通解释。**经过 MLP、LayerNorm 和门控后的分支表示，不再保证只保留原谱频段**；需单独测量。

### 5.3 真正与 v2 不同的设计重点：按状态“选择是否注入”，而不是把两路都做强

输入投影仍然保持：

\[
H_{raw}=\operatorname{Linear}_{1\to128}(x).
\]

两个分支独立编码：

\[
U_L=\mathrm{MLP}_L(L_2),\quad
U_H=\mathrm{MLP}_H([B_0,B_1]).
\]

建议 `MLP: C→16→128`、中间 SiLU，末端使用轻量归一化或控制输出标准差；只在一开始选择一种，并保持与参数对照一致。

门控输入为当前状态与局部条件：

\[
G_i=[x_i,\log(1+d_i),\log(1+c_i),\log(1+r_i),\operatorname{clip}(\Delta H_i),\eta_i],
\]

其中 `η_i` 可以取由 `[B₀,B₁]` 计算得到的局部尺度差强度。**第一阶段不加入搜索时间 t**；原 `generate_data()` 只保存状态、不保存其产生的时间。如果将来加时间/停滞信息，必须同步修改训练数据和推理接口，不能仅推理时增加输入。

使用**两个独立 sigmoid 门控**：

\[
g_L=\sigma(f_L(G)),\quad g_H=\sigma(f_H(G)),
\]

而非 `g_H=1−g_L`。允许同时关闭、同时开启两路。再加入**可靠的残差保护**：

\[
\boxed{
H_0=H_{raw}+\alpha_L\,g_L\odot U_L+\alpha_H\,g_H\odot U_H
}
\]

其中 `α_L, α_H` 建议从小幅非零值 `0.005~0.02` 的范围做验证集选择，默认先试 `0.01`；不要把它们当作固定最优超参。与 v2 一样保留可学习系数，但这次**单独记录其演化、门控条件差异、输出范数及最后 logits 的实际改变**。

**必要零扰动测试**：当两个残差系数都被置零时，新模型应与原权重给出逐元素一致的 logits（允许浮点零误差），对应可复现的原 RLNN 结果。这是防止“代码重构本身改了 Baseline”的单元测试。

### 5.4 ASCII 结构图（v4-A，PAVE 保持原样）

```text
 输入：无向图 G=(V,E)，K 条动态候选解 X_t [N,K]
                      |
           +----------+-----------+
           |                      |
           v                      v
   原 RLNN：Linear(1,128)   图状态描述 Z_t [N,K,C]
           |                    [x, log1p(c)]
           |                      |
           |                      v
           |            固定拓扑 P = 0.5(I+D^-1 A)
           |                      |
           |                 Z_1 = P Z_t
           |                 Z_2 = P Z_1
           |                      |
           |            +---------+----------+
           |            |                    |
           |            v                    v
           |       LOW 分支                HIGH 分支
           |       L2 = Z2                 B0 = Z-Z1
           |       较平滑信号              B1 = Z1-Z2
           |            |                  多尺度差异
           |            v                    |
           |       MLP 16→128               v
           |            |               MLP 16→128
           |            |                    |
           |            +--------+-----------+
           |                     ^
           |                     |
           |       条件门控 [x,c,degree,r,ΔH,band]
           |          gL=Sigmoid, gH=Sigmoid
           |                     |
           |                     v
           +----> H0 = Hraw + αL*gL*UL + αH*gH*UH
                                 |
                                 v
                     原五层残差 GCN (128维)
                                 |
                                 v
                     原 Linear(128,1) -> logits
                                 |
                                 v
                  Bernoulli 翻转，更新 X_(t+1)
                                 |
                    重复 K=20, T=100 次
                                 |
                                 v
             原能量档案 -> A.3 解码，得到合法解
                                 |
                                 v
                     [PAVE-Stats H1：冻结、原样]
                                 |
                                 v
                   最终合法独立集 Size / Time
```

### 5.5 为什么“频域接在原 GCN 之前”仍需审慎？

五层残差 GCN 已有邻居传播和自身路径。**从表达能力上**，原网络可以在多层中隐式利用类似局部平滑与差异，不能说它“一定完全丢失高频”。v4-A 的价值需要来自**更容易优化的频段参数化、状态条件门控或新的有效特征**，而非“以前没有邻居信息”。

因此必须做两个控制：

1. 同样 `Z`、同样门控、同等参数量的**非频域 MLP/普通消息分支**；
2. 同样低高频特征，关闭门控或使用固定融合的版本。

如果非频域控制同样好，严格结论应是“额外状态编码有用”，不是“频域模块有独立贡献”。

---

## 6. 结构升级：v4-B 任务条件图频率，仅在 v4-A 确有收益后开发

v2 的 `exchange_band` 是任务相关的邻域差分，但没有标准谱解释。若后续确实需要**由搜索状态改变滤波拓扑**，推荐采用**对称、非负的状态依赖边权**，构造有谱定义的动态图算子。

### 6.1 先定义被唯一阻挡的边

对当前候选状态 `x`，`c_j=∑_{i∈N(j)}x_i`。对原图每条无向边 `i−j`，定义：

\[
\rho_{ij}(x)
=\mathbf1[(x_i=1,x_j=0,c_j=1)\lor(x_j=1,x_i=0,c_i=1)].
\]

该条件关于 `i,j` 对称，表示一条连接“已选节点”与“只被一个已选节点阻挡的未选节点”的关系。**对一般不合法 x，这仍然只是局部条件，不保证全图合法交换。**

动态权重：

\[
W_{ij}(x)=A_{ij}[\epsilon+\gamma\rho_{ij}(x)],\quad
\epsilon>0,\gamma\ge0.
\]

于是 `W(x)` 对称、非负。定义动态度矩阵 `D_W(x)`，

\[
P_x=\frac12(I+D_W(x)^{-1}W(x)).
\]

这时每个固定 `x` 对应一个明确的可归一化谱滤波算子。但由于 **x 每步更新、20 条轨迹的 x 又不相同**，`P_x` 也按轨迹/时间变化。不可把 20 条轨迹混在同一张动态邻接图上。

### 6.2 动态分频与静态分频怎么融合？

- **静态分支**：`P_static` 在整个图上固定，捕捉普遍局部占用与冲突传播。
- **动态分支（可选）**：`P_x` 针对唯一阻挡关系，计算 `Z−P_x Z`，反映状态相关的局部差异。
- **条件融合**：只有当验证表明动态差分对后续翻转收益有贡献时才接入，否则继续使用轻量 v4-A。

推荐首先只让动态分支输出**一个小维度残差**，不直接替代原五层 GCN 的图结构。这样仍可保留基础表示与默认更新策略。

### 6.3 为什么这条路线要谨慎？

- 状态相关边权在每个推理步、每条候选轨迹都会变化，必须用按边的 `[E,K]` 聚合实现；不能复用静态 CSR 的乘法结果。
- 计算复杂度至少包含每步 `O(|E|K)` 的资格判断与归一化；两次滤波还会增加 `O(|E|KC)`。v2 已经显示 2 倍级的时延风险。
- **唯一阻挡关系不等于最终改进方向**。v3 证明中性平台的后续搜索更重要，因此动态谱分支的输出应针对后续质量验证，而不是以重新识别唯一阻挡边作为唯一训练目标。
- 为避免错误的频率解释，必须验证 `W=Wᵀ`、非负权重、归一化一致性和孤立节点边界。没有这些条件不能宣称谱定理直接适用。

**建议决策**：只有在 v4-A 的低频/高频分支消融能显著提升验证解质量、且动态关系提供超出原静态 P 的新增信息后，才考虑 v4-B。否则这个方向的成本大于研究价值。

---
## 7. 训练设计：不要让新模块仍然只学到容易学的局部能量

### 7.1 第一阶段先不改损失，隔离频域表示的价值

原训练逻辑保留：

\[
q_i=x_i(1-p_i)+(1-x_i)p_i,
\]

\[
\mathcal L_{\mathrm{erdoes}}
=\beta\sum_{\{i,j\}\in E}q_iq_j-\sum_iq_i
+\lambda\left(\sum_i p_i-d\right)^2.
\]

这里的 `p_i` 仍是**条件独立 Bernoulli 翻转概率**。若只改变 GCN 前端、输出仍为独立 Bernoulli，原期望能量计算可继续使用。若以后改成**联动翻转、相关采样**，原 `q_iq_j` 的独立期望假设未必成立，不能直接沿用。

**这一阶段的核心检验**：在相同主目标下，分频能否提高：
- 单图搜索中的 best-energy 曲线；
- 解码前冲突数与有效局部调整；
- 最终 A.3 后 Size；
- 冻结 PAVE 后的最终 Size；
- 0.5 秒预算下的最终 Size。

**不需要**为了“看起来有创新”立刻增加多个新损失；先把已有变量控制好。

### 7.2 为什么原损失可能与频域收益不对齐？

原损失重视下一步低能量，然而 PAVE 验证的改善主要来自中性合法动作以后继续搜索。对于 MIS：

- 单节点能量有利的方向，不必然形成有效的多节点交换；
- 高频差异强不等于未来收益高；
- 单步能量下降可以发生，但解码后的 Size 不一定增加；
- 为获得非局部改善，搜索可能需要暂时不降能量的路径。

因此如 v4-A 在单步主损失下继续无收益，**先用离线诊断确认是否存在频域可预测的多步收益**，再决定是否引入额外监督。**没有离线正证据，不要先增加 loss 使实验难以解释。**

### 7.3 可选的第二阶段：面向非平局动作的辅助排序（只作为候选改进）

在**训练集**中抽取原 RLNN 搜索产生的难状态（尤其搜索中后期停滞、局部冲突频繁的状态），只对少量候选进行配对试探：

1. 固定同一图、同一状态、同一后续搜索算法；
2. 使用共享随机数比较两个候选动作 `a,b` 的有限继续收益；
3. 只保留实际收益不同的动作对作为监督；
4. 在可比动作规模下，将翻转 logits 聚合为候选评分 `s(a)`；
5. 在普通 erdoes loss 外，用很小权重加入成对排序损失：

\[
\mathcal L_{\mathrm{pair}}=\log\bigl(1+\exp(-[s(a^+)-s(a^-)])\bigr).
\]

注意：
- 这不是修改 v3 PAVE 的头或其训练！它只是在**RLNN 频域模块的训练阶段**可选增加辅助信号；
- 辅助动作排名 `s(a)` 是启发式训练代理，不等于实际 Bernoulli 采样会一次执行整个联合动作；
- 若动作为不同节点数，需等大小比较或明确归一化，不可让候选大小混淆收益；
- 若标签仍几乎全为 0（v1 FlipValue 曾出现此问题），则应停止该辅助损失方向；
- 绝不能用作者测试集生成监督，避免测试泄漏。

建议只有在**验证集的非平局动作排序探针**能显著优于普通消息/统计 MLP 时，才真正训练该辅助损失。它是研究备选，不属于推荐第一版的必需部件。

### 7.4 训练预算必须对齐

建议两阶段、同种子、相同更新数：

| 阶段 | 比较对象 | 训练设置 | 目标 |
|---|---|---|---|
| A0 数值/结构冒烟 | 原 RLNN vs 零扰动 adapter | 不训练、少量图 | 验证完全兼容与正确性 |
| A1 短期适配筛选 | 原 RLNN 微调 vs 参数匹配 MLP vs 普通消息 vs v4-A | 冻结主干 1 epoch + 解冻 2 epoch；同 LR/状态/更新数 | 快速判断是否有必要扩大试验 |
| A2 充足训练验证 | 筛选通过的 2—3 个候选及对应对照 | 50 epoch、训练与验证协议统一；3 个训练种子 | 判断是否稳定提升，不再只看一个 seed |
| A3 独立结构消融 | low / high / two-band / gate / release / scale / time | 根据 A2 的最佳共同方案逐项关闭 | 证明是**哪一部分**有效 |
| A4 可选动态拓扑 | v4-A vs v4-B，同参数/时间 | 仅 A2 成功后启动 | 衡量状态改变图滤波器是否值得 |

**预检选择只看验证集**，完整测试集只用于确定方案后的最终报告，避免用 1000 张测试图反复挑结构。

---

## 8. 实验矩阵：每组回答一个明确的问题

### 8.1 主消融（优先级按顺序，不必一次全部跑完）

| ID | 方案 | 对照目的 | 预期判断方式 |
|---|---|---|---|
| B0 | 原 RLNN（冻结已复现最佳权重） | 绝对基线 | 固定 K/T 与计时协议 |
| B1 | 原 RLNN + 同参数 raw-state MLP adapter | **容量对照** | 频域不能只赢原主干，至少要优于 B1 |
| B2 | 原 RLNN + `[Z,PZ]` 普通消息 adapter | **同信息/邻域对照** | 隔离“滤波坐标变换”的效果 |
| F0 | v4-A 静态低频（只 `L2`） | 低频是否足够 | 同参数控制 |
| F1 | v4-A 近邻差分（只 `B0`） | 高频单独贡献 | 同参数控制 |
| F2 | v4-A 多尺度残差（`B0+B1`） | 二阶差异有没有价值 | 与 F1 比较 |
| F3 | v4-A 双分支固定 0.5 | 频带并用是否有效 | 与 F0/F2 比较 |
| F4 | **v4-A 双分支独立门控** | **主要候选** | 比 F3、B1、B2 都应更好才有意义 |
| F5 | F4 去掉 `r` | 释放代理是否真的有用 | 差异仅归因于 `r` |
| F6 | F4 去掉 `ΔH` | 能量先验贡献 | 是否只是给网络显式能量梯度 |
| F7 | F4 单门控（`gH=1−gL`） | 独立门控是否必要 | 看原设计竞争是否限制信息 |
| F8 | F4 α 固定 vs 可学习；禁用归一化 | 融合稳定性 | 查看梯度/激活是否异常 |
| F9 | v4-B 动态图滤波 + 同等轻量分支 | 状态图拓扑是否值得 | **只在 F4 通过后做** |

**关键否定性对照**：F4 如果只优于 B0，却与 B1/B2 持平或更差，则**不能认定高低频设计有独立优势**。若它仅在参数更多、搜索更久的情况下提高 Size，则也没有通过当前研究门槛。

### 8.2 模型评价必须分两张表

**表 A：频域模块本身（不接 PAVE）**

| Model | Train Seeds | Params ↓ | Mean Size ↑ | Δ vs B0 ↑ | Search Time ↓ | Feasible Rate |
|---|---:|---:|---:|---:|---:|---:|
| RLNN | 3 | ... | ... | — | ... | ... |
| MLP capacity control | 3 | ... | ... | ... | ... | ... |
| Ordinary-message control | 3 | ... | ... | ... | ... | ... |
| v4-A | 3 | ... | ... | ... | ... | ... |

**表 B：固定 v3 PAVE 后的完整管线**

| Pipeline | PAVE Head | Mean Size ↑ | Δ vs RLNN+PAVE ↑ | 0.5s Size ↑ | Total Time ↓ |
|---|---|---:|---:|---:|---:|
| RLNN + PAVE | 固定 Stats H1 | ... | — | ... | ... |
| MLP control + PAVE | 同一个固定 Stats H1 | ... | ... | ... | ... |
| Message control + PAVE | 同一个固定 Stats H1 | ... | ... | ... | ... |
| **v4-A + PAVE** | **同一个固定 Stats H1** | ... | ... | ... | ... |

只看表 B 不能判断模块是否真正改善了 RLNN 的翻转策略，因为 PAVE 后处理会改变收益。必须同时给出表 A。

### 8.3 必测诊断（比再增加 5 个模型更重要）

| 指标 | 具体记录什么 | 能排查什么问题 |
|---|---|---|
| Raw energy / conflict curve | t=0,10,25,50,75,100；每条轨迹 best energy、冲突边数 | 模块是在改善搜索还是只改变最终解码 |
| Logit change | `logit_new − logit_baseline` 的绝对值、分位数、按度数/状态分层 | adapter 是否真在改变动作 |
| Actual expected flips | `∑_i p_i` 的均值、分布 | 新模块是否扰动原 d=5 目标 |
| Band reconstruction error | `||Z−(L2+B1+B0)||` | 实现是否满足频带定义 |
| Band energy statistics | 按搜索阶段统计 low/high 来源信号及分支输出范数 | 是否发生尺度崩溃或无效放大 |
| Gate distribution | `gL/gH` 的直方图、分位数、对状态/冲突的条件分布 | 是否只是常数门控 |
| Adapter contribution | `||αL gL UL||/||Hraw||`、`||αH gH UH||/||Hraw||` | 增量过大还是过小 |
| Ablation drop | 禁用 low/high/gate 时的最终 ΔSize | 分支是否真实有用 |
| Pairwise action utility | 同状态共享随机数的收益差、排序 AUC/相关性（只在非平局子集） | AUC 是否与**未来收益**相关 |
| Post-PAVE gain | 对同一冻结 PAVE，分别记录 PAVE 前后 Size 与耗时 | 是否因后处理掩盖频域收益 |
| GPU cost | 稀疏 mm、MLP、门控、缓存/构图、解码、PAVE 各阶段 wall-time | 收益是否被计算开销抵消 |
| Cross-size behavior | RB-small 过关后，在新图规模/ER 检查 | 是否只适合已有图分布 |

### 8.4 公平实验口径

- **测试集**：作者公开两组 `idx_0..999`，严格按 ID 有序评估；Mean Size 按逐图整数结果计算，不用 fp16 对整数直接聚合。
- **训练种子**：至少 `0,1,2`，区分训练种子与搜索种子；同模型/同图固定 K/T，逐图配对。
- **定量**：三种子的 `Mean ± sample Std`，同时报单图配对胜/平/负和配对 `ΔSize`；同一图跨种子不能被当成独立的 3000 张图。
- **时间**：同型号空闲 RTX 5090，暖机、CUDA 同步；包括每步频域计算、采样、解码、PAVE 的完整实际求解时间。定时任务单卡同时只跑一个评测进程。
- **两种预算**：固定 `K=20,T=100` 和固定 `0.5 s/图`，另给 `Size vs Wall-time` 曲线。固定时间必须返回截止前已验证的合法 incumbent，记录超时数、实际平均步数。
- **参考值**：RLSA、KaMIS 作为外部强方法；没有逐图证明最优时，不能凭 KaMIS 的结果计算真正的 optimal gap。
- **对 PAVE 的约束**：**模型权重、特征、动作池、继续深度 H1、选择准则保持不变**；如果新 RLNN 产生不同分布的起点，需单独报告“PAVE 额外改进量”和分布变化，不把其收益算作频域直接贡献。

### 8.5 预注册式停止条件

建议在跑 1000 图测试前写入实验计划，避免事后选择：

1. **失败停止**：三种子验证均不超过同信息 Message/Stats 对照，或超过但总时间明显变差并在 0.5 秒预算内无收益 → 停止当前结构。
2. **继续研究**：至少两个种子在验证集显示明确改善，负控组不重复这一改善，且额外时间可接受 → 再提交完整测试。
3. **独立创新暂定成立**：正式测试中，F4 相比 B2 在**相同计算时间**下也存在稳定的质量提升；分支禁用会降低质量；相关诊断解释与真实收益一致。没有三项证据，仍称“候选方法”，不要夸大为成立的创新。

这里不给出人为承诺的 `+0.2`、`+0.5` 性能数字。需要由验证结果决定是否值得继续。

---

## 9. 工程改动清单与单元测试

### 9.1 建议新增文件，暂不动 v3 PAVE

```text
RLNN_lab/
├── models/
│   ├── mis_v4_frequency.py          # 静态/动态频段、轻量门控、adapter
│   ├── diagnose_v4_frequency.py     # 频率/门控/轨迹/成本诊断
│   ├── models.py                    # 原 GCN 继续保持，必要时仅暴露中间特征
│   └── solvers.py                   # 只添加 frequency='v4a/v4b' 接口；原采样不动
├── train_scripts/
│   └── v4_frequency/
│       ├── README.md                # 冻结实验协议
│       ├── train.sh                 # 从同一 checkpoint + 同一预算训练
│       ├── eval.sh                  # 未接 PAVE 的测试
│       ├── eval_with_fixed_pave.sh  # 接固定 v3 PAVE
│       └── diagnose.sh             # 零扰动、分支指标、耗时
├── outputs/v4_frequency/
│   └── <run_name>/                  # run.json, per_graph.csv, train.csv, diagnostics
└── tools/
    └── analyze_v4_frequency.py     # 配对统计 / 图表 / 拆分耗时
```

**本次只提供方案，不声称上述文件已经创建或服务器已经执行。** 对 `train_scripts/legacy/v3/`、`models/mis_v3_search.py`、`models/mis_v3_value.py` 和 v3 PAVE checkpoint **不作任何修改**。

### 9.2 前向伪代码（说明接口，不是已运行的补丁）

```python
# x: [N, K]，图之间可合批为块对角邻接，K 维彼此独立。
# A_binary: 无向二值图；adj_energy 的边权仍为原 MIS 的 1/2。

def frequency_adapter(x, A_binary, static_cache, model):
    c = graph_neighbor_sum(A_binary, x)              # [N,K]
    d = static_cache.degree                            # [N]
    release = ((x == 0) & (c == 1)).float()
    r = x * graph_neighbor_sum(A_binary, release)     # 仅“释放”代理

    z = torch.stack([x, torch.log1p(c)], dim=-1)     # [N,K,2]
    pz = lazy_walk(z, static_cache)                   # [N,K,2]
    ppz = lazy_walk(pz, static_cache)
    low = ppz
    high = torch.cat([z - pz, pz - ppz], dim=-1)

    gate_stat = build_gate_features(x, c, d, r, low, high)
    g_low, g_high = model.two_gates(gate_stat)

    delta = (model.alpha_low * g_low * model.low_mlp(low)
             + model.alpha_high * g_high * model.high_mlp(high))
    return delta


def flip_logits_v4(x, G, model, cache):
    raw = model.original_initial_project(x.unsqueeze(-1))
    h0 = raw + frequency_adapter(x, G.adj_binary, cache, model.adapter)
    return model.original_five_layer_gcn_and_output(h0, cache.gcn_adj)
```

实际需要支持训练/测试不同的 `K`、混合精度与任意图批。每步中的 `c,r,low,high` 必须依赖当下 `x` 重新计算；**只能缓存图的静态拓扑**。

### 9.3 非常容易出错的数值/张量边界

1. **`K` 不是特征宽度**：不要固定输入 `[N,20]`；训练还可能是 `[N,64]` 或 `[N,1000]` 的状态批。
2. **权重 1/2**：能量邻接的每个无向边双向记录各乘 1/2，计算原始度数和局部邻居选中数时要恢复二值邻接，否则 `c`、`degree`、`release` 全错。
3. **孤立节点**：`d=0` 时规定滤波保留自身，`high=0`；`r=0`，不能生成 NaN。
4. **同名图不同边**：缓存不能只用 `N` 作 key，必须绑定图身份、边和设备；每步动态状态不缓存。
5. **FP16 Sparse MatMul**：不要假设 fp16 CSR 的所有操作在目标 torch/CUDA 版本上都支持；优先对滤波在 FP32 中计算，再与模型激活按策略转换。
6. **正负方向**：高通差分包含正负，不能先全程只取 ReLU 导致丢失负向差异；确要使用双符号通道，需对照其计算开销与表达收益。
7. **可学习 α 初始化**：α=0 会让分支内部参数在初始步接不到梯度；建议小的非零初始化并单独验证。
8. **静态频带 / 动态关系**：不要拿动态图差异的数值直接代入静态 `L_sym` 的频谱公式解释。
9. **训练/评估流程**：新的 `--frequency` 选项不能影响 baseline 的默认 checkpoint 加载或生成状态随机流。
10. **CPU/PAVE**：PAVE 的 CPU 后处理仍照旧，求解器计时包含 CPU↔GPU 同步等实际开销，不能把它排除出新模型总时间。

### 9.4 必须先跑完的测试用例

| 测试 | 预期 |
|---|---|
| 全零/全一状态 | 每个常数通道的高通差分为零；孤立点不出错 |
| 小图手算 | `Z≈L2+B1+B0`，误差在浮点容差内 |
| 节点重编号 | 图、节点状态一起置换后，logits 只相应置换 |
| 候选轨迹重排 | 不同 K 列不会串扰，输出按列重排 |
| 多图块对角 | 合批结果与逐图计算一致，梯度同预算一致 |
| 关闭 adapter | 恢复原 RLNN logits、Size 统计和相同随机流 |
| 训练梯度 | 两个分支、门控、α 有有限且非零梯度（可学习时） |
| 前向/反向显存 | 没有随 1000 状态一次展开导致不必要峰值显存 |
| 耗时覆盖 | 所有图滤波、GCN、解码、PAVE 后处理均计入 |
| 最终合法性 | 对输出节点集合独立检查冲突边数为 0 |
| v4-B 对称权重（如实现） | `W=W.T`、`W>=0`、每个轨迹正确构造 |

---

## 10. 研究路线与阅读次序

### 10.1 第一周建议只做“诊断 + 小模块”，不要直接完整训练

**任务一：** 读 GRACE、WaveGC、HeroFilter、COPT-MT、ICLR 2025 Spectral-benchmark 的方法图、门控/谱公式和限制；确认哪个机制**与 RLNN 已实现内容重合**。配合源码对照。

**任务二：** 从 v2 诊断出发扩展日志：在 `t=0,25,50,75,100`，统计每条轨迹的 `x,c,r,ΔH`、band energy、gate、残差、是否出现可收益的动作；区别“识别规则候选”与“预测收益”。

**任务三：** 实现 v4-A 的**零扰动适配**和 B1/B2 普通 MLP/消息对照，只在训练/验证集上做短期筛选。记录三组在相同迭代预算下的 Size 和每步时间。

**任务四：** 如果有效，再做动态门控、低/高分支等细消融；无效就收缩模型，记录失败原因，不继续扩大频率专家数量。

### 10.2 推荐的文献阅读顺序（从最重要到补充）

1. **ICML 2026 COPT-MT / LoG 2025 GCON**：最直接的 MIS + 多滤波器先例，决定新颖性上限。
2. **NeurIPS 2025 GRACE**：低通/高通专家 + 节点门控已有多成熟，避免重复设计。
3. **ICLR 2025 Spectral-benchmark**：如何检查你是真的“利用频率”，而不是仅名字上有高低频。
4. **ICML 2025 WaveGC**：具体多尺度图滤波与尺度残差如何构造。
5. **NeurIPS 2025 HeroFilter / AAAI 2026 MORGAN**：若确需更灵活谱响应，再学习条件选择与专家分工。
6. **CVPR 2024 FADC / TPAMI 2024 FreqFusion / CVPR 2026 F2Net**：迁移你擅长的分支特征处理、门控与训练诊断，但不能照搬二维模块。
7. **ICML 2024 SLOG / NeurIPS 2024 S²GNN**：考虑更大尺度或归纳泛化时补读。

### 10.3 如何判断最后能否写成论文中的一个独立创新点？

必须能回答下面四个问题：

1. **Motivation**：不是“GCN 丢高频”，而是“当前搜索状态中哪些约束相关的尺度差异没有被有效利用”，有日志证据。
2. **Mechanism**：为什么使用当前状态、约束信息来条件选择多尺度图滤波；与 GRACE、GCON、WaveGC 的具体区别是什么。
3. **Effectiveness**：为什么这个分频机制比**同信息同参数的普通消息模型**好；没有这个对照就不能成立。
4. **Efficiency**：同样总时间下是否仍提高合法 MIS Size；如果只是更慢、略微更好，不足以支撑“高效求解”的定位。

若四点都无法满足，应如实把频域内容降为**消融对照或负结果**，把论文主线放在已经有真实正收益的 PAVE（但这一轮按你的决定不修改 PAVE）。

---

## 11. 最后结论与下一步实际执行决策

**结论 1：现有低高频设计没有解决明确的搜索短板。** v1 主要是对已有邻域统计做一阶变换；v2 虽做了多尺度与交换条件差分，但收益与耗时不匹配；v3 的频率评分头也没有超出相同特征的普通消息/统计版本。

**结论 2：下一版不应“更深、更多频带、更复杂门控”优先。** 最合理的结构是轻量多尺度图分频 + 状态条件独立门控 + 小残差注入原 RLNN，同时保留参数匹配的非频域强对照。它是一项可验证的重设计，不代表一定创新或一定涨点。

**结论 3：任务条件信息必须真正在搜索中有意义。** 对局部约束、唯一阻挡和释放机会的理解要与“能否得到更大的合法独立集”区分；`exchange_band` 的候选 AUC 高不能直接证明高频有用。若要用动态交换图，先给出数学上明确的图滤波器并评估每步开销。

**结论 4：PAVE 保持 v3 已验证版本**，所有新模型分别报告**不加 PAVE**与**接同一个冻结 PAVE**的结果，不把后处理提升混入频域模块贡献。

> **一句话行动建议：** 先开发 **v4-A：任务状态双尺度低/高频 adapter**，与“同信息同参数的普通消息 adapter”做最严格的对照；只有在验证集显示明确解质量收益、且总耗时可控时，才开发 v4-B 的动态任务图频率。若失败，及时停止继续堆频域结构。

---

## 附录：本次调研的可追溯来源与限定

### A.1 RLNN_lab 源码/实验记录

- [RLNN v1 分频与任务统计](https://github.com/YuqiWang-code/RLNN_lab/blob/63a353e282fcbcda58b46302fa1ff8f85e802bd3/models/mis_v1.py)
- [RLNN v2 频域结构](https://github.com/YuqiWang-code/RLNN_lab/blob/63a353e282fcbcda58b46302fa1ff8f85e802bd3/models/archived_frequency/mis_v2_frequency.py)
- [v2 机制诊断脚本](https://github.com/YuqiWang-code/RLNN_lab/blob/63a353e282fcbcda58b46302fa1ff8f85e802bd3/models/archived_frequency/diagnose_v2_frequency.py)
- [v2 SEGF 诊断 JSON](https://github.com/YuqiWang-code/RLNN_lab/blob/63a353e282fcbcda58b46302fa1ff8f85e802bd3/outputs/v2/diagnostics_frequency_segf/run.json)
- [v2 实验协议](https://github.com/YuqiWang-code/RLNN_lab/blob/63a353e282fcbcda58b46302fa1ff8f85e802bd3/train_scripts/legacy/v2/README.md)
- [v3 统计/消息/频域评分器源码](https://github.com/YuqiWang-code/RLNN_lab/blob/63a353e282fcbcda58b46302fa1ff8f85e802bd3/models/mis_v3_value.py)
- [v3 最新实验分析](https://github.com/YuqiWang-code/RLNN_lab/blob/63a353e282fcbcda58b46302fa1ff8f85e802bd3/docs/temporary/RLNN_v3_Results_Analysis.md)
- [v3 训练标签与结果汇总](https://github.com/YuqiWang-code/RLNN_lab/blob/63a353e282fcbcda58b46302fa1ff8f85e802bd3/outputs/v3/summary.txt)
- [v3 固定工作量逐种子对照](https://github.com/YuqiWang-code/RLNN_lab/blob/63a353e282fcbcda58b46302fa1ff8f85e802bd3/outputs/v3/analysis/fixed_tuned_h1_summary.csv)

### A.2 调研边界

- 文献综述基于**2024—2026 正式论文页面、公开摘要/方法描述、作者仓库与可访问原文**；所列 PDF 和 GitHub 是阅读/下载入口，并不表示全部论文都已在本项目机器上复现。
- 现有实验数字来自 GitHub 在 `63a353e` 提交中保存的真实 `run.json/CSV` 和项目汇总；**本次没有修改仓库代码，也没有训练或测试新网络**。
- 每个“可能有效”的判断均属于**后续需要对照验证的假设**，不得在论文摘要中提前写成已经证明的贡献。
- v4-A、v4-B 为研究代号；如果实验不显示独立收益，应修改或放弃其创新定位。
