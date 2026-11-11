# RLNN Lab v7：置换稳健的一次性 Mamba 静态结构先验——代码改造与完整实验方案

> **版本**：v7 研究/实施设计稿（2026-11-10）  
> **仓库**：https://github.com/YuqiWang-code/RLNN_lab  
> **本次审查基准**：`main`，commit `99e5b73a73bac6c7fe208693772d40c1205fa18f`（2026-11-09 11:58 UTC）  
> **任务**：RB200–300 图上的 Maximum Independent Set（MIS）；RLNN 为随机二值状态翻转搜索器，A.3 将候选状态解码为合法独立集，PAVE-Stats H1 在合法解上做平台搜索。  
> **状态**：设计方案，**尚未修改仓库、安装 Mamba、训练 v7 或取得 v7 指标**。文中的性能数字仅来自仓库中已完成的历史运行；阈值、预期和实验排期是预注册建议，不是新实验事实。

---

## 0. 先给出最终决定

**停止沿 v4/v5/v6 的频域架构继续堆分支；不再替换 RLNN 的 5 层残差 GCN；不修改、重训或重新挑选 PAVE-Stats H1。**

v7 只研究一个可证伪的命题：

> 在迭代式二值状态搜索中，**一次性**计算、**近似置换一致**的静态全图结构上下文，能否以足够低的在线开销为原 RLNN 提供超出廉价局部统计/普通消息编码的增量决策信息，并最终提高**同预算合法独立集大小**？

推荐名称：**PC-OSM-RLNN**（Permutation-Consistent One-Shot Mamba for RLNN）。论文中不要把“新颖性”写成“首次将 Mamba 引入神经组合优化”“Mamba 缓解 GCN 过平滑”或“Mamba 比 Transformer 更快”；这些都不是当前仓库的实证问题，也有明显先例。更合理的贡献应是：

1. **One-shot graph context**：图结构只编码一次，在 `T=100` 或更长的采样阶段 **不再调用 selective scan**；20 条粒子轨迹共享同一节点上下文。
2. **Action-asymmetric prior**：静态全图上下文不是直接替代骨干，而是预测每个节点“未选中时被翻转”和“已选中时被翻转”的**两个不同偏置**，与当前二值状态发生极廉价的点式交互。
3. **Permutation and information controls**：同输入特征、近似同参数的 MLP/普通消息/随机排序控制，以及图重编号后的输出与最终 Size 稳定性检验。
4. **Budget-first evidence**：区分“单步/固定 100 步指标”与“包含结构预处理、Mamba 编码、A.3、冻结 PAVE 的 0.5/1.0 秒整图求解”；不以模型容量或显存指标代替最终搜索收益。

**主线只允许一个 Mamba 插入位置：原 GCN 最终 logits 后、Bernoulli flip 前。** PAVE 的 24 维统计特征、3,713 参数评分网络、H1 中性动作宏、训练权重和推理配置全部维持原样。

---

## 1. 仓库核查、已有证据和真实风险

### 1.1 已核查的关键路径

已扫描默认分支的文件树（3,073 个条目，含大量历史输出/快照），深入查看与本方案直接相关的实现和结果：

- `models/models.py`：`GCN` 是 5 层残差结构，`initial_project -> [GCNConv + Linear + ReLU + residual] × 5 -> final_project`。原始 GCN 参数 165,505。
- `models/solvers.py`：模型工厂、`RLNN.flip_logits()`、`policy_topology()`、`sample_states()`、Erdős 期望能量与期望翻转数训练。多图训练每个 graph batch 内还会按状态列分片并**多次 optimizer.step()**。
- `models/main.py`：CLI 入口；已有 `--frequency` 属历史兼容字段，不应再让 v7 语义绑定频域。
- `models/eval_v2.py`、`models/eval_v3.py`：同图同随机流搜索、A.3、合法性、PAVE、wall-clock，固定步数与截止时间的口径不同。
- `models/capture_mis.py`：CUDA Graph 捕获一次采样 step 并反复 replay；这决定了 Mamba **必须在捕获前**预计算，否则会随 step 重放。
- `models/train_v4_frequency.py`、`models/train_v5_frequency.py`：冻结基线教师生成 matched states、v5 从强 B 权重热启动、3 epoch/6,000 updates 的训练入口。
- `models/mis_v3_search.py`、`models/mis_v3_value.py`：合法平台搜索与 PAVE-Stats H1；**仅作为冻结下游使用，不进入修改范围**。
- `models/mis_v6_frequency.py`、`models/train_v6_frequency.py`、`models/diagnose_v6_frequency.py`、`outputs/v6/analysis/pilot.json`：v6 架构与完整第一轮 No-go 结果。
- `docs/temporary/RLNN_v3_Results_Analysis.md`、`RLNN_v4_Frequency_Mechanism_Followup_Results.md`、`RLNN_v5_Frequency_Results.md`、`RLNN_v6_SC_RLNN.md`：历史结论与实验边界。

**注意**：这是针对本次迭代的完整“代码路径—训练—评测—证据”审查，不声称逐行人工阅读了 3,073 个条目的所有生成结果。`models/data/` 的大型训练数据与部分权重未在 GitHub 公开，正式执行仍需服务器已有缓存和检查点。

### 1.2 必须保留的强基线

| 证据/协议 | 已完成的仓库结果 | 含义 |
|---|---:|---|
| v3：相同合法起点，统计 PAVE H1 vs rule23 | `+0.06033` Mean Size，作者测试集 1,000 图 × 3 种子 | PAVE 有真实学习收益，继续冻结使用 |
| v5：B 持续搜索 + PAVE，0.5 秒 | `19.6767`，100 验证图 × 3 种子 | v7 限时必须打的强对照 |
| v5：B 持续搜索 + PAVE，1.0 秒 | `19.8300`，100 验证图 × 3 种子 | 继续搜索比额外逐步网络更有性价比 |
| v5：DAFR + PAVE，0.5/1.0 秒 | `19.5000 / 19.6167` | 增加一步代价会显著压缩搜索收益 |
| v6：B + A.3，FP16、K20/T100、CUDA Graph | `19.5267 ± 0.0929` | v6 新协议下强骨干的实际参考值 |
| v6：SC-RLNN + A.3 | `17.4433 ± 0.0416` | 从零改骨干后的质量崩溃，下降 `2.0833` |
| v6：B + A.3 + PAVE | `19.6567 ± 0.0850` | 与同协议 v6 比较的最终参考值 |
| v6：SC-RLNN + A.3 + PAVE | `19.0733 ± 0.0493` | PAVE 可补救部分质量，但仍落后 B `0.5833` |
| v6：已训练 seed42 的 FP16 配对测速 | SC/B = `1.3846×` | 速度不满足 1.08× 准入 |

**不要混用这些数字**：v3 的官方作者测试、v5 的时间预算验证和 v6 的 FP16/CUDA Graph 固定工作量，来自**不同协议**。v7 的每组正式对照都要在同一套参数、精度、图 ID、随机数和独占计时口径下重测 B。

### 1.3 从 v6 得到的工程教训

- v6 的每层独立 local/neighbor 投影和读出改造并非简单“小补丁”：它从随机初始化重新训练整个网络，首筛仅 1 epoch/2,000 updates；**不能仅凭此认定“全局上下文必然无效”**，但足以支持 v7 不动骨干。
- v6 的极化正则/跨层差分并没有保证最终 Size；第五层 Dirichlet 比和高带 logit 能量也未达门槛。**表征变化不等于动作有效**。
- v6 中 `log_degree /= log_degree.max()` 在多图合批训练时用**整个块图的最大值**，单图推理时用单图最大值，存在可核查的训练/部署特征尺度不一致风险。v7 静态特征一律使用**固定训练集归一化统计**或每图定义严格一致的公式，禁止按 graph batch 最大值归一化。
- `run.json` 中 v6 `source_sha` 为 null。v7 需要把 commit SHA、完整模型/环境配置、数据 manifest 哈希、权重哈希和随机流全部锁定，否则回归分析会失去可靠性。

---

## 2. v7 主结构：GCN 不变，Mamba 一次性编码，两个状态依赖偏置

### 2.1 前向流程

```text
                         静态图 G = (V, E)
                                  |
                 [一次性静态特征 φ(G)，不含 x]
                                  |
             [结构序列化 π，含同分组随机扰动]
                                  |
           [共享权重双向 Mamba，小宽度 d=32]
                                  |
                   [逆置换回原节点 ID]
                                  |
                       [节点嵌入 g_i]
                                  |
                [Linear(d,2) -> tanh -> 缓存]
                                  |
                         (b_add, b_drop)
                                  |
         +------------------------+--------------------+
         |                                             |
         |              不随 t 和粒子 k 变化            |
         v                                             v
  图 G + 当前 x^(t,k)                         每个粒子共享 b_i
         |                                             |
 [原 5 层残差 GCN，原权重]                                |
         |                                             |
    logit_B(i,k,t)                        x_i=0 -> b_add(i)
         |                                 x_i=1 -> b_drop(i)
         +--------------------------+------------------+
                                    |
                          小幅、零初始化残差
                                    |
                         logit_v7(i,k,t)
                                    |
                    Bernoulli flip / 能量档案
                                    |
                         A.3 合法解码
                                    |
                   **原始冻结 PAVE-Stats H1**
                                    |
                         最大独立集输出
```

**一次性**指每张图、每个完整 `search()` 调用只运行 Mamba 前向一次；不是声称“加入静态先验后每步毫无算术开销”。每个 step 仍有非常小的点式 `where/add`，必须实测。

### 2.2 为什么选“两动作偏置”而不是直接拼接 128 维 hidden

MIS 采样动作的语义由当前状态决定。同一个节点：

- 若 `x_i=0`，翻转是**加入**，与邻接冲突风险相关；
- 若 `x_i=1`，翻转是**移除**，与释放邻域及后续可修复性相关。

把全图结构压成一个不区分状态的单一 logit 常量容易只学到固定节点偏好；让 Mamba 每步读 `x` 则又回到 v4/v6 的逐步成本困境。**两个静态动作先验 + 动态二值选择**是折中点。

推荐主式：

\[
\begin{aligned}
  B_i(G) &= \tanh\big(W_o\,g_i(G)+b_o\big)
                  = [b_i^{\mathrm{add}}, b_i^{\mathrm{drop}}],\\
  r_{ik}^{(t)} &= (1-x_{ik}^{(t)})b_i^{\mathrm{add}}
                     +x_{ik}^{(t)}b_i^{\mathrm{drop}},\\
  \ell_{ik}^{v7}(t)&=\ell_{ik}^{B}(t)+
            0.50\tanh(\alpha)\,r_{ik}^{(t)}.
\end{aligned}
\]

- `alpha` 是**唯一**最终残差门幅参数，初始化为 0；因此 v7 **epoch0 的 logits 与 B 一致**（fp16 输出前必须将残差转为 B 的 dtype）。
- `W_o` 保持非零随机初始化，`b_add/b_drop` 有非零变化；首步 `alpha` 能拿到梯度，`alpha` 打开后 Mamba 才逐渐收到梯度。
- `tanh` 限制 `|delta_logit| ≤ 0.5`。这只是主设置，可在训练图上预先校准 `0.25/0.5/1.0` 的灵敏度；**不看测试集挑系数**。
- 不重复计算 `A @ x`、不新增 SpMM、不引入粒子之间的交互。基线 GCN 的原 5 层、输入 `x`、隐层维数 128、图归一化以及优化目标均可原样保留。
- PAVE 仍以完成的合法初始解为输入；**不把 Mamba embedding 接到 PAVE**。

**预留消融**：若两个静态动作偏置表现过弱，再尝试一个不新增稀疏传播的轻门控：

\[
\delta\ell_{ik}(t)=0.50\tanh(\alpha)\,r_{ik}(t)
\cdot\sigma(a\ell_{ik}^{B}(t)+b),
\]

这是可选 `v7_mamba_logit_gate`，不是首轮主线。它能使用当前状态下已有的 GCN logit 而不额外计算 `Ax`。不能一开始混入主实验，否则难区分 Mamba、门控和参数量的作用。

### 2.3 静态输入特征，必须不含动态搜索状态

首轮推荐 8–10 维，按“廉价、可复算、有全局对照价值”排序：

| 静态特征 | 说明 | 成本与实验控制 |
|---|---|---|
| `log1p(degree)` | 度数的稳定尺度 | 基线可用局部统计；必须进入所有控制 |
| 邻居平均 `log1p(degree)` | 一跳度分布 | 每图构造一次 |
| 邻居 `log1p(degree)` 标准差 | 局部结构异质性 | 每图构造一次 |
| `k-core` index | 结构壳层 | 一次 CPU 图遍历；与 MLP/MSG 共享 |
| 到高优先级 hub 的 BFS 最短距离 | 跨节点位置 | **同分 hub 必须随机打破平局**，计入前处理 |
| 2-hop 可达节点数 / N | 两跳覆盖度 | 去重计数；时间计入一次性结构预处理 |
| 全图密度 `2m/(n(n-1))` | 图级可比较条件 | 对所有节点广播，控制组相同 |
| `n/300` | 图规模信息 | 控制分布外变化 |
| 可选 clustering/triangles | 局部闭三角 | 若 0.5 秒计时不划算，直接删除并归档 |
| 可选 degree percentile | 图内相对度数 | 只在证明有效后启用 |

工程规定：

1. 首先只实现**廉价 8 维**（前八项）；三角形和聚类系数暂不进主线。
2. 对数、标准化、clamp 和缺失值规则固定；用训练集统计 `mean/std`，写进 checkpoint，验证和测试绝不能重新拟合。
3. RB 图中高度数/高 k-core 并不必然意味着“该节点应该被加入”。Mamba 负责学结构条件；**不要用度数大小直接伪造监督标签**。
4. 当静态特征中含 hub/BFS 时，hub 的选择也是算法的一部分；不能先用原节点编号选 hub，再声称模型置换等变。
5. 静态结构处理、排序与 Mamba 运行的耗时，都必须纳入线上计时。若实验假设图会被重复查询，可另报 `amortized` 延迟，但**不能用摊销值替代主 0.5/1.0 秒指标**。

### 2.4 节点结构序列化与置换一致性

**默认结构序**：

1. 在图上取 hub 候选（按度、k-core 的稳定可计算优先级；同分选取按独立随机流均匀采样）。
2. 从 hub 做 BFS 分层；主要按 BFS 距离、k-core、degree 排序。
3. 在结构评分完全相同的 bucket 内用单独的 `order_seed` 随机打散，而非节点 ID 决胜。
4. 取得 `order` 后执行双向共享权重 scan，并逆置换回原节点：

```python
z = input_projection(static_features)              # [N, 32]
s = z[order][None, :, :]                            # [1, N, 32]
fwd = mamba(s)
bwd = mamba(s.flip(dims=(1,))).flip(dims=(1,))
h_seq = layer_norm(s + 0.5 * (fwd + bwd))
h = torch.empty_like(h_seq[0])
h[order] = h_seq[0]                                 # 恢复真实节点顺序
biases = torch.tanh(output_projector(h))            # [N, 2]
```

- **共享权重双向**优先于两个独立 Mamba：减少参数，同时对序列整体反转具有天然的对称平均形式；仍**不保证**任意图节点置换等变。
- 推理默认**1 个结构序**。若单序置换抖动大且时间余量允许，再做 `S=2/4` 条同 bucket 随机顺序的平均，作为**一次性图预处理**的一部分。
- **绝不**把 20 条粒子或 `T=100` 时间步串成一条 Mamba 序列；这会混淆独立粒子并破坏计算预算。
- 空图/孤点/不连通图必须定义 BFS 未访问值和多连通分量规则。对于多连通分量，要么分别 scan 再拼回，要么对每个分量显式重置 SSM；**不允许跨不相连图或跨 graph batch 泄漏 hidden state**。
- 对 graph batch 中 8 张图，首版**逐图独立**运行 Mamba，明确以正确性优先；之后可改为合法 padding/segmented scan。若使用 padding，反向扫描必须先截取有效长度再翻转，不能让无效 padding 进入有效节点的历史。

### 2.5 Mamba 的推荐超参

| 项目 | 首轮设置 | 后续允许的变化 |
|---|---|---|
| Mamba | `mamba_ssm.Mamba`（Mamba-1 模块） | 不同时引入 Mamba2/Mamba3，避免架构混杂 |
| `d_model` | `32` | `16` 作为省时消融；64 只在明显收益时尝试 |
| `d_state` | `16` | `8` 为省时消融 |
| `d_conv` | `4` | 固定 |
| `expand` | `2` | 固定 |
| block 数 | **1** | 不在 v7 主线堆层 |
| scan 方向 | 共享权重前向+反向 | 单向是必要消融 |
| 图排序 | BFS/degree/core 结构序 + tie jitter | degree sort、全随机 shuffle 对照 |
| 推理 order 数 | 1 | 稳健性触发后再比较 2/4 |
| 输出 | 每节点 2 个标量 | 无新增中间状态 GCN 分支 |
| 对原 GCN | **完全冻结**为主实验 | 单独的低学习率骨干微调为附加消融 |

这一架构的参数会高于 B，参数不能预填估计值；统一使用 `sum(p.numel() ...)` 记录。对照 MLP/MSG 要以**实际参数 ±5%**匹配并披露差异，且共享相同静态输入。

---

## 3. models/ 应该如何修改：精确到文件、接口和关键代码

### 3.1 新增文件与职责

```text
models/
  mis_v7_context.py           # V7StaticContextGCN、Mamba/MLP/MSG 适配器、零启动残差
  v7_structure.py             # 静态图特征、per-graph 结构序、relabel 映射、图分量
  train_v7_context.py         # matched-state 训练 + 冻结 B + v7 记录
  eval_v7_context.py          # 复用 eval_v3 的 fixed/budget + CUDA Graph 一次性缓存
  diagnose_v7_context.py      # permutation / feature redundancy / logit / per-state 健康诊断
  profile_v7_context.py       # one-shot prepare、T100、A.3、PAVE 分段独占计时
  probe_v7_context.py         # 训练图 proxy/弱标签可用性与成本预检
train_scripts/v7/
  README.md
  common.sh
  check.py                    # CPU/CUDA、batch、置换、warm start、梯度与 cache 自检
  p0.sh                       # 零训练 preflight
  train.sh                    # 单臂、单种子、3 epochs
  eval.sh                     # 100 图 fixed work
  eval_budget.sh              # 0.5/1.0 秒 limit
  profile.sh                  # 独占测量
  diagnose.sh                 # P0/P1 诊断
  decide.py                   # 多门槛、图级 bootstrap、No-go
outputs/v7_context/
  preflight/
  state_manifest/
  train_*/
  eval_*/
  budget_*/
  profile_*/
  diagnosis_*/
  analysis/
```

**不改**：`models/models.py` 中 GCN 的数学定义、`models/mis_v3_search.py`、`models/mis_v3_value.py`、PAVE 的三个 `best.pt`。尽量沿用现有 `models/capture_mis.py`，只通过统一拓扑/策略接口让一次性缓存自然生效。

### 3.2 `models/main.py`：新增独立 CLI，而非继续借 `--frequency`

建议加入：

```python
parser.add_argument(
    '--context_mode', default='none',
    choices=('none', 'v7_mamba', 'v7_mlp', 'v7_message',
             'v7_shuffle_mamba', 'v7_static_linear', 'v7_mamba_logit_gate')
)
parser.add_argument('--v7_width', type=int, default=32)
parser.add_argument('--v7_d_state', type=int, default=16)
parser.add_argument('--v7_order_seed', type=int, default=0)
parser.add_argument('--v7_order_views', type=int, default=1)
parser.add_argument('--v7_delta_limit', type=float, default=0.5)
parser.add_argument('--v7_state_source', default='teacher',
                    choices=('teacher', 'mixed'))
```

保留历史 `--frequency` 不变；若 `context_mode != none`，强制 `frequency == none`。**禁止将 v7 分支再命名为 v7_frequency**，避免实验结果和代码语义继续混在旧项目里。

### 3.3 `models/v7_structure.py`：拓扑对象约定

建议用不可变 NamedTuple 或 dataclass 保存：

```python
V7Topology(
    normalized,         # 原 edge2adj，GCN 不变
    static_features,    # [sum_N, F], float32，固定训练集统计归一化
    graph_ptr,          # 每张图独立区间，示例 (0, 238, 490, ...)
    orders,             # 每张图、每个 view 的局部节点次序
    cached_action_bias, # None（训练）或 [sum_N, 2]（推理预计算）
)
```

- 单图构造直接 `graph_ptr=(0, N)`；合批时从 `graph.batch` 连续区间得到边界，**不可**把 8 张图整体送成一条序列。
- `normalized` 使用原 `edge2adj(graph.edge_index, graph.num_nodes)`，数值一致性须验证。
- `static_features` 在图首次进入设备时构造，静态图属性不得读 `x`、当前翻转概率、PAVE 最终 Size 或测试集标签。
- 排序中的同分随机数从独立 `order_generator` 产生，绝不能更改搜索的 `torch.Generator(seed=1017+i)` 状态。
- `cached_action_bias` 训练时为 None；推理时预计算，整段搜索持有。**不写跨模型、跨 checkpoint、跨精度、跨 graph 的全局缓存**；多次独立求解默认各自计费一次，避免污染延迟口径。

### 3.4 `models/mis_v7_context.py`：关键模型骨架

以下是应遵守的**核心逻辑示意**，静态特征计算和 MLP/MSG 模块需按上一节接口实现后才能组成可运行完整文件；不要当作已经在仓库执行过的成品：

```python
import torch
from torch import nn
from models import GCN

class V7ActionPrior(nn.Module):
    def __init__(self, mode, feature_dim, width=32, d_state=16):
        super().__init__()
        self.mode = mode
        self.in_proj = nn.Linear(feature_dim, width)
        self.norm = nn.LayerNorm(width)
        self.out_proj = nn.Linear(width, 2)
        self.alpha = nn.Parameter(torch.zeros(()))   # EXACT B at epoch0
        if mode in ('v7_mamba', 'v7_shuffle_mamba', 'v7_mamba_logit_gate'):
            from mamba_ssm import Mamba  # lazy import，其他控制不依赖 CUDA 扩展
            self.sequence = Mamba(d_model=width, d_state=d_state,
                                  d_conv=4, expand=2)
        elif mode == 'v7_mlp':
            self.sequence = ...  # 参数匹配的静态 node MLP + graph mean
        elif mode == 'v7_message':
            self.sequence = ...  # 参数匹配的 2-hop 静态消息编码
        else:
            self.sequence = None  # v7_static_linear

    def precompute(self, topology):
        # 每个 graph_ptr 区间、每个 order view 独立计算。
        # 对于 Mamba: seq -> forward/reverse -> invert order -> mean views
        # 对于 MLP/MSG: 同一 static_features，非序列化对应控制。
        # 输出 float32、形状 [N, 2]，且没有动态 x。
        return ...

    def correction(self, x, bias, delta_limit=0.5):
        # x: [N, K]，bias: [N, 2]，0=加入；1=移除
        x32 = x.float()
        node_bias = (1.0 - x32) * bias[:, 0:1] + x32 * bias[:, 1:2]
        return delta_limit * torch.tanh(self.alpha) * node_bias

class V7StaticContextGCN(GCN):
    def __init__(self, hidden, layers, mode='v7_mamba',
                 feature_dim=8, width=32, d_state=16, delta_limit=0.5):
        super().__init__(1, hidden, layers, 1)
        self.adapter = V7ActionPrior(mode, feature_dim, width, d_state)
        self.delta_limit = delta_limit

    def forward(self, x, edge_index, energy_adj, topology=None):
        # topology 必须提前由 solver.policy_topology() 生成。
        if topology is None:
            raise RuntimeError('v7 requires an explicit topology')
        base = super().forward(
            x.unsqueeze(-1), edge_index, topology.normalized
        ).squeeze(-1)  # 原 B 路径、原数值运算保持不变
        bias = topology.cached_action_bias
        if bias is None:  # train: 每个 optimizer update 重建带梯度的图
            bias = self.adapter.precompute(topology)
        delta = self.adapter.correction(x, bias, self.delta_limit)
        # 避免 fp16 zero-residual 时将 B 无意中提升到 float32。
        return base + delta.to(base.dtype)
```

实现重点：

- `out_proj` 必须有**非零**初始化；**只**把 `alpha` 初始化为 0。若 `alpha` 和 `out_proj` 同时置零，梯度可能同时被封死。
- 对 Mamba 主臂，`precompute()` 先按每图拆分，进行 `in_proj`、一次双向 scan 和逆置换，再用 `out_proj(...).tanh()` 得到 `[N,2]`；所有控制使用一致的输出范围。
- `alpha` 可以使用独立的较大学习率，encoder 的 learning rate 单独设置，**记录第一步 alpha 梯度、打开后的 Mamba 梯度、`|delta_logit|` 与 `|delta_prob|`**。
- `V7StaticContextGCN` 继承 `GCN` 的键名，原 B 的 `initial_project`、`convs`、`linears`、`final_project` 名字不变；热启动时只允许缺少 `adapter.*`，原 B 不允许 unexpected keys。
- 在 fp16 中做 Mamba 数值诊断时优先保证 `bias` 有限，返回 logits dtype 与 B 一致。不要用 CPU 隐式 fallback 宣称是 GPU 加速。
- 任何静态 Mamba hidden 的全局 max/mean 归一化，都应明确按**单张图**操作；训练 graph batch 不能跨图归一。

### 3.5 `models/solvers.py`：四处必须改对

**A. 模型选择**：

```python
from mis_v7_context import V7StaticContextGCN
from v7_structure import v7_topology

self.v7_context = getattr(args, 'context_mode', 'none') != 'none'
if self.v7_context and self.frequency != 'none':
    raise ValueError('v7 context cannot be combined with legacy frequency')
if self.v7_context:
    self.model = V7StaticContextGCN(
        args.num_h, args.num_l, mode=args.context_mode,
        feature_dim=V7_FEATURE_DIM, width=args.v7_width,
        d_state=args.v7_d_state, delta_limit=args.v7_delta_limit)
else:
    # 保留原来的 none / V1-V6 分支
    ...
```

**B. 热启动**：扩展现有 `strict=not (...)` 分支：v7 可以 `strict=False`，但必须断言 `unexpected_keys == []` 且**所有** `missing_keys` 都以前缀 `adapter.` 开头。按同 seed 从 `outputs/v2/none_seed{0,1,2}/checkpoints/best.pt` 加载 B。**不要用 v6 权重、不要从随机初始化训练 B**。

**C. 拓扑与缓存**：

```python
def policy_topology(self, graph, energy_adj):
    if self.v7_context:
        topo = v7_topology(graph, energy_adj, ...)
        if not self.model.training:  # eval path, before search / capture
            model = self.accelerator.unwrap_model(self.model)
            with torch.no_grad():
                bias = model.adapter.precompute(topo)
            topo = topo._replace(cached_action_bias=bias)
        return topo
    # old branches unchanged
    ...
```

**关键**：`topology` 由同一个 `search()` 先构造，随后贯穿 100 或更多 step。不能在 `model.forward()` 遇到 None 就每步重新计算 Mamba。正式评测直接拒绝缺少预先构造 topology 的调用。

**D. flip logits**：

```python
def flip_logits(self, x, graph, adjacency, normalized=None):
    if self.v7_context:
        return self.model(x, graph.edge_index, adjacency, normalized)
    # else unchanged
```

由于 `sample_states()` 已经把 `normalized` 持续传入 `flip_logits()`，这样 v7 能复用原采样器、原 Bernoulli、能量档案、A.3 和冻结 PAVE。

### 3.6 最容易忽略的 bug：训练中不能直接缓存带梯度的 Mamba 输出

当前 `RLNN.train_graph_batches()` 对同一个 graph batch 的状态列分多个 `batch_size=64` 分片，**每个分片都会反向传播并进行一次 `optimizer.step()`**。因此：

- 推理：`precompute()` 在 `torch.no_grad()` 下运行一次并保存 `[N,2]`，完全正确。
- 训练：**不能**先算一次带梯度 `g`，在随后的 16 个独立 optimizer step 中反复使用同一计算图。会出现 `Trying to backward through the graph a second time`，或因为权重已更新而使用过时前向。
- 首版可靠做法：静态特征与排序每 graph batch 只算一次；**带梯度的 Mamba forward 每 optimizer step 重新计算一次**。这增加训练成本，不影响部署设计。
- 更高级做法：一个 graph batch 的所有状态分片先累计梯度，最后**只 step 一次**；必须匹配 Erdos 损失加权、显存和有效优化器更新数，不能声称仍与 v5 的 2,000 updates/epoch 完全同口径。

**首轮建议使用第一种**，保证训练定义与仓库最接近；汇报训练吞吐时诚实记录 Mamba 每个 update 的开销。推理“one-shot”不等于训练“one-shot”。

### 3.7 `models/capture_mis.py`：不允许把 Mamba 捕进 step Graph

现有 `captured_states()` 在 `torch.cuda.graph(captured)` 中记录 `step()`，然后 `captured.replay()` 100 次；如果 static encoder 写在 `model.forward()` 内，**Mamba 也会被录入 CUDA Graph 并每步重复执行**，彻底违背设计。

修复方式不是去重写整个 `capture_mis.py`，而是遵守上一节接口：`policy_topology()` 在调用 `captured_states()` **之前**就生成 `cached_action_bias`；捕获的 `step()` 只包含原 GCN 和 `where/add`。

必须在 profiler 中观察：

- 一图 `Mamba.forward / selective_scan` 次数应为 `2 × order_views`（双向，按图一次）；
- 100 次 graph replay 时 selective scan 调用次数 **为 0**；
- 不同图、不同求解任务正确重新预计算；同一次图的 20 条轨迹不触发重复扫描。

**精确性能提醒**：CUDA Graph 记录、首次 kernel/JIT、warm-up 是否计费，必须与 B 的现有协议一致；单独报告编译/捕获冷启动和正常求解常态，不得只把 Mamba 计算挪到“计时起点之前”。

### 3.8 `models/train_v7_context.py`：建议独立训练器而非复制 v6 训练器

从 `train_v5_frequency.py` 的**热启动 + 冻结 B + matched teacher** 协议派生，而不是复制 `train_v6_frequency.py` 的随机初始化。推荐训练阶段：

**阶段 A（主实验，完整 3 epoch）**

- `seed/reference_seed`: `(42,0)`、`(123,1)`、`(999,2)`；或使用统一 0/1/2，但必须固定配置、保持映射清楚。
- 每组从相应 B 检查点 warm start。
- `GCN` 全部冻结 3 epoch，主臂只更新 `adapter.*`。
- `lr_encoder=3e-4`、`lr_out=3e-4`、`lr_alpha=1e-3`；Adam/AdamW 选择先固定一种（若沿用原 Adam，建议主实验统一使用 Adam），如需 weight decay 对所有控制一致。
- `1000` 张训练图、`100` 张验证图、`graph_batch_size=8`、`batch_size=64`、`num_k=20`、`num_tp=50`、`num_t=100`、`num_l=5`、`num_h=128`、`num_d=5`、`beta=1.02`、`lambd=0.5`、`loss=erdoes`、`mixed_precision=no`；`3 epoch × 2000 updates = 6000 updates`。
- 学生和控制从同一冻结教师 B 的轨迹状态开始。排序 jitter **不能消耗或打乱** teacher/搜索的 RNG，必须单独维护并持久化随机流。
- 先不要加入 v6 的 `polarization_bonus` 和 `orthogonality`，也不要同时追加新的 rollout 奖励；否则结果不是单纯的静态上下文机制效果。
- 选权重使用验证集固定工作量的 `A.3 Size`，不使用训练 loss 或 PAVE 对本次模型的单种子偶然补偿来选模型。

**阶段 B（有证据才启用）**

- 若静态先验能改善固定步数，但总体仍差，可尝试固定种子、训练图上的 20% 学生 on-policy 状态混合，比较 `teacher_only` 与 `mixed_80/20`；需要各控制共享可复现的状态缓存。
- 若纯 adapter 收益为正而受限，再单独以 `1e-5` 微调 B，必须加入 **同预算 B-only fine-tune** 对照；不是 v7 首轮的默认操作。
- 可选 loss：`L = L_erdoes + λ_KL KL(p_B || p_v7)`，仅在首轮出现大幅 policy drift 时作为附加臂。`λ_KL` 在训练图内预注册；不把辅助 loss 伪装成无条件有效。

**固定教师状态一致性**：当前 `MatchedStateRLNN.generate_data()` 直接使用默认全局随机流；不同臂的 order jitter 会改变后续教师轨迹。建议为 teacher 状态、DataLoader shuffle、order jitter、搜索 Bernoulli 分别设独立 `torch.Generator`；更稳的是对 `(train_seed, epoch, graph_id)` 确定性生成，并为前 20 图记录状态 hash，验证各臂完全一致。否则“matched states”可能只是配置相同而非逐状态相同。

### 3.9 `models/eval_v7_context.py`：沿用 eval_v3，减少口径漂移

- 可参照 `eval_v6_frequency.py`，但把检查切到 `context_mode`，保留 `eval_v3.evaluate()` 的完整工作量与 wall-clock 两个入口。
- 所有 fixed work：`K20/T100`、`num_d=5`、`beta=1.02`、A.3、`post=none/pave`、`horizon=1`、PAVE `candidate_cap=32`、`exchange_rounds=3`、`outer_rounds=8`、`improving_cap=48`、`repair_rounds=4`、`post_reserve_s=0.05`。
- 评测强制同一个 `split=validation` 的 100 张图、相同搜索随机流 `1017 + i`、相同 checkpoint 选权重策略，记录 `graph_id` 与最终节点集合。
- 时间预算评估必须传 `--extended-search`，允许 B/v7 在预算内持续采样；绝不能让 B 走超过 T100 而让 v7 固定 T100，或者反过来。
- 同一个 B 的独立固定时间求解与 v7 的固定时间求解虽可用相同种子，但实际截止时刻、轨迹长度不同，必须报告**整套配置差**，不能宣称严格相同起点的因果差。
- 固定工作量的 `none/pave` **共享同一个初始解**，用于识别 PAVE 后处理增益；不同模型的同图同种子才用于整体 policy 比较。

### 3.10 `models/utils.py` 与数据部分

- 若 `build_batch()` 生成 `Data.batch`，为 v7 补 `graph_ptr`（节点偏移列表）更稳，不要在 GPU 上反复 `unique()`/`.cpu()` 求区间。
- 单图无 `graph.batch` 时按 `[0,N]` 处理。
- 如需建立图结构特征缓存，用 `graph_id + edge_hash + feature_schema_version + normalization_sha` 作 key，并在实验中区分“在线一次性处理”和“离线预处理已摊销”两种口径。
- 输入数据训练/验证/作者测试图 ID 顺序应锁定 manifest，防止多进程 glob 次序影响配对。

---

## 4. P0：不盲目开大训练的先证伪实验

> **纠正一个重要推理错误**：拿一个**未训练、随机权重** Mamba 产生的 `g_i`，用局部统计回归出高低 R²，并不能证明它“有用/没用”。低 R² 可能仅代表随机非线性不可拟合；高 R² 可能是统计维度低。也不能在 `alpha=0` 的零残差条件下用最终 Size 做置换敏感性结论——输出必然等于 B。P0 必须把“构造正确性”“结构信息增量”“下游有用性”和“完成训练后的置换稳健”区分开。

### P0-0：基础正确性（不通过直接修复，不进入训练）

运行 `train_scripts/v7/check.py`：

1. **B 权重继承**：逐项比较 B 与 v7 的原 backbone tensor，所有 `GCN` 权重完全一致；只允许缺少 adapter 键。
2. **零残差回归**：固定图、x、随机权重、原 B 最终 logits，测试 CPU FP32、GPU FP32、FP16：`alpha=0` 时 v7 logits 与 B 数值一致；A.3 最终结果同一搜索随机流一致。严格浮点逐位检查用于同精度同执行模式，跨不同 CUDA kernel/FP16 原生重复只能给容差和轨迹统计。
3. **起步梯度**：第一步 alpha 梯度非零；第二步 encoder/out projector 梯度有限且可非零；排除双重零初始化。
4. **图批不串扰**：两图分别编码的结果与块对角图、按 graph_ptr 分开编码结果一致；单图复制/粒子扩充不会改变静态输出。
5. **节点重编号**：无排序 tie 的人工图中按逆置换还原后输出一致；含 tie 的图测分布与方差（不能要求单个随机顺序精确等变）。
6. **零/单/孤立/不连通图**：没有 NaN、索引越界、BFS 未访问值污染或跨分量 scan 串扰。
7. **CUDA Graph 缓存**：selective scan 只在预处理触发，100 次 replay 中不触发。
8. **三份 PAVE 权重核对**：记录 `sha256`，与 v3/v5 历史记录保持一致。

输出 `outputs/v7_context/preflight/{cpu,cuda,zero,graph_batch,permutation,cuda_graph,manifest}.json`。

### P0-A：真实成本门槛（必需；优先于任何 6000-update 训练）

只需随机初始化 Mamba 即可检查**结构成本上限**，因为前处理的计算图形状/算子与训练后基本一致；但最终还是要对已训练 checkpoint 重测。

- 20 张训练图，不用验证集挑速度；RTX 5090 独占，B 与零残差 v7 分别交错运行，5 次重复，FP32 与 FP16 + CUDA Graph 各一组。
- 记录 `static_features_ms`、`ordering_ms`、`mamba_scan_ms`、`topology_ms`、`capture_ms`、`T100_rollout_ms`、`A3_ms`、`PAVE_ms`、`full_ms`、P50/P95、每图搜索步数。
- 将 Mamba 前处理放入搜索计时 `tick` 之后；检查是否有 CPU 同步、排序 Python 循环、反向扫描 padding、`.item()` 引起的隐形开销。
- **预筛建议**：T100 + A.3 的整图时间比 B **≤1.08×**；允许由于小样本计时抖动进入“复测”区间 `1.08–1.12×`，但不允许直接当作通过。若 `>1.12×` 且计时稳定，先优化 one-shot 实现/feature/scan order，再决定是否终止。
- 额外测更贴近真实目标的 `0.5s`/`1.0s` 同预算，因为 static overhead 会挤压可继续搜索的步数。此时 `alpha=0` 的质量没有研究意义，只检验工程速度和 baseline 一致性。

### P0-B：有意义的“冗余/可用信息”检验（小成本）

**B1 结构统计可恢复性（解释性探针）**

- 用训练图中不同采样状态 `t∈{25,50,75,100}`，每图抽相同数量的节点/粒子，基线输入集为 `[x, neighbor_count, degree, neighbor_occupancy, ΔE, B 原 GCN hidden(或 logits)]`；可按成本另测一跳/两跳静态局部特征。
- 目标不是未训练 `g`，而是 **实际静态输入的各个全图结构特征**，例如 k-core/BFS 层、2-hop coverage 等。分别做 Ridge/小 MLP，报告逐特征 held-out `R²` 或分类准确率。
- **按 graph_id 分 train/probe split**，不可在同一张图节点上随机划分训练和验证后宣称能泛化。
- 高 R² 表明该特征在当前局部表征下容易估计，但**不能单独触发 No-go**；应进入 B2 的下游增量判定。

**B2 静态结构对“好解”是否有条件增量（真正的 Go/No-go 依据）**

- 在少量**训练图**中对每图多个独立随机起点运行冻结 B/A.3/可选冻结 PAVE，收集不同合法解。构造**代理标签**：节点在高质量独立解集合中的被选频率、或可重复的入选偏好；明确它**不是最优 MIS 真值**，也不是最终 Size 保证。
- 模型 1：廉价局部静态特征 + 小 MLP；模型 2：模型 1 + 全局静态特征；模型 3：模型 2 + 学习得到的 Mamba 上下文；模型 4：同参数 MSG/DeepSets 上下文。**相同训练图、相同标签、相同参数预算、跨图留出**。
- 看跨图 held-out 的 AUC/相关性/校准与增量（不只看 train R²）。在同标签上做 graph-level bootstrap；模型 3 必须至少优于便宜的 1/2，且相对 4 有趋势，才支持进入主实验。
- 可重用仓库的 `probe_v4_utility.py`、`probe_v4_utility_followup.py` 检验候选动作收益标签的重复性，但 v4/v5 曾观察到两半 Spearman 约 0.15、非平局率弱，**不能**把这些噪声标签当成监督可靠的金标准。
- 如果 Mamba 只比“没有全局输入”的廉价模型好，却不比**同信息的 MSG/MLP**好，说明收益来自额外静态信息而非 Mamba 扫描；这种结果只支持“静态结构先验”，不支持“必须用 Mamba”。

**B3 反事实小探针（可选）**

- 在训练图上挑高质量候选与少量 t25/50/75/100 状态，给每个节点先验偏置施加极小固定 perturbation，使用共同随机数重放少量步，比较平均最终合法 Size 与差值方差。
- 报告零增益比例和置信区间，若不可辨，**放弃把 P0-B3 当训练标签**，并非否定原始 Erdos 训练。

### P0-C：重编号敏感性与排序基准

对相同的训练图做 3–5 个节点随机重编号（保持同构），排序种子固定/变化两组都跑：

- `alpha=0` 时：检查 B logits 对齐一致、v7 缓存的 `bias[N,2]` 的节点对齐差、随机结构序本身的变化；**不以 Size 波动给 Mamba 下结论**。
- 为观察 scan 自身稳定性，可暂时设固定非零小 `alpha`，**此时只报告诊断，不报告模型性能改善**；或者等短训后在固定 checkpoint 上进行正式指标。
- **公平共同随机数**：对图重编号的同时，必须把每一步随机矩阵 `U[node, particle, step]` 按同一置换同步重排。否则同一个 RNG 种子只是生成相同线性数组，分配给了不同节点，Size 差可能仅来自 Bernoulli 随机流重绑定。
- 对比原 B 的自然 FP16 不确定性；FP32 用确定性或固定执行模式作结构诊断，FP16 报原生重复波动，不能把核级差异归为序列化问题。
- 计算 `bias` 对齐后 MAE、cosine、相对方差、top-q 节点顺序变化；训练后才比较 A.3 Size、PAVE Size 的同图分布与图级 bootstrap。

### P0 进入训练的判断

**硬门槛**：正确性全部过关、每步没有 Mamba、静态成本可接受。**信息门槛**：至少有一种跨图代理标签显示全局输入在廉价控制之外存在稳定方向；不能因为随机 Mamba 的 `g` 难被线性回归而直接通过。

如果成本超限，先减掉三角形/多视图/CPU 同步、使用 `d_model=16`；如果同信息静态 MLP/MSG 已经覆盖信号，**即使 Mamba 很酷也建议 No-go**。若 P0 下游代理标签本身不可靠，则允许做一次极小 1-seed/1-epoch 的真实 Size pilot，但不得直接开全矩阵。

---

## 5. P1：规范化三种子训练与同参数对照

### 5.1 实验臂与主要可识别问题

| Arm | 骨干 | 静态信息 | 额外网络 | 排序 | 识别问题 |
|---|---|---|---|---|---|
| `B` | 原 v2 B，冻结 | 无 | 无 | 无 | 最强原始政策 |
| `B+FT` | 相同 B 微调控制 | 无 | 无 | 无 | 区分预算/微调收益（仅附加阶段） |
| `V7-Linear` | 相同 B | 8 维 | 线性/浅 MLP | 无 | 便宜统计是否已足够 |
| `V7-MLP` | 相同 B | **完全相同 8 维** | 参数匹配的 DeepSets/MLP+全图均值 | 无 | 无序全局统计是否足够 |
| `V7-MSG` | 相同 B | **完全相同 8 维** | 一次性两跳消息编码，参数匹配 | 无 | 额外消息是否足够 |
| `V7-Mamba`（主臂） | 相同 B | **完全相同 8 维** | 1×共享双向 Mamba | BFS 结构序 | 测试 SSM 顺序选择性 |
| `V7-Mamba-Shuffle` | 相同 B | **完全相同 8 维** | 相同 Mamba | 全随机序，**独立训练** | 结构序是否必要 |
| `V7-Mamba-Degree` | 相同 B | **完全相同 8 维** | 相同 Mamba | degree 序，**独立训练** | BFS 相比静态度排序的贡献 |
| `V7-Mamba-Uni` | 相同 B | **完全相同 8 维** | 相同 Mamba 单向 | BFS 结构序 | 双向扫描的边际收益 |

**核心矩阵不要一开始全部跑**。先按资源优先级 `B`（复用已冻结权重并同协议重测）、`V7-MLP`、`V7-MSG`、`V7-Mamba`、`V7-Mamba-Shuffle` 共 5 臂 × 3 种子。`Linear` 可先单 seed，`Degree/Uni/2-view` 只在主臂表现正向后增加。对 v7 各臂训练 3 epoch/6000 updates，数据、teacher、随机流严格配对。

**对照完整性**：

- 普通 MSG 的一次性传播同样在图搜索前做完，**不得**让 MSG 每步做 SpMM 而 Mamba 只做一次，否则失去公平性。
- 静态特征、two-action residual、alpha 初始化、训练损失、教师轨迹、输出范围、PAVE、预算相同，主要差异只有编码算子/排序。
- DeepSets 的全图均值、2-hop 信息必须计入算子和参数成本；不能只用不包含全图信息的小线性层当所谓“同信息对照”。
- 参数总数、实际 trainable params、FLOPs（可选但非唯一度量）、全部在线延迟都记录；不要以“同参数”隐去操作数差异。

### 5.2 固定训练设置（预注册）

```yaml
problem: MIS / RB200-300
train_graphs: 1000
validation_graphs: 100
held_out_author_test_graphs: 1000
teacher_backbones:
  ref0: outputs/v2/none_seed0/checkpoints/best.pt
  ref1: outputs/v2/none_seed1/checkpoints/best.pt
  ref2: outputs/v2/none_seed2/checkpoints/best.pt
teacher_reference_seeds: [0, 1, 2]
student_train_seeds: [42, 123, 999]
train:
  epochs: 3
  updates_per_epoch: 2000
  graph_batch_size: 8
  state_batch_size: 64
  num_k: 20
  num_tp: 50
  precision: fp32
  beta: 1.02
  erdoes_lambda: 0.5
  target_flips_d: 5
  optimizer: Adam
  encoder_lr: 3.0e-4
  head_lr: 3.0e-4
  alpha_lr: 1.0e-3
  backbone_frozen: true
  teacher_states: frozen_B
  order_seed_stream: independent
  use_polarization_loss: false
  use_frequency_loss: false
eval:
  fp32_primary: true
  fp16_secondary: true
  num_k: 20
  num_t: 100
  search_seed_start: 1017
  A3_decoder: unchanged
  PAVE: outputs/v3/stats_h1_seed{0,1,2}/checkpoints/best.pt
  PAVE_frozen: true
  horizon: 1
  candidate_cap: 32
  exchange_rounds: 3
  outer_rounds: 8
  improving_cap: 48
  repair_rounds: 4
  post_reserve_s: 0.05
  time_budgets_s: [0.5, 1.0]
  extended_search_when_budgeted: true
```

**与 v5 的区别要说清**：v5 的 adapter 112 参数、首轮冻结后解冻骨干；v7 因 Mamba 体量更大，主实验推荐**3 轮一直冻结 B**。因此 v7 的“6000 updates”只是状态训练预算对齐，**不是**与 v5 完全同样的可训练参数设置。若论文要公平证明方法收益，增设同参数/同算子控制和必要的 `B+FT`。

### 5.3 主训练应记录的量

每次 update/epoch 至少记录：`erdoes_energy`, `expected_flips`, `distance_penalty`, `alpha`, `delta_logit_rms`, `delta_prob_mean_abs`, `alpha_grad`, `mamba_grad_norm`, `static_bias_std`, `b_add/b_drop_mean`, `train_seconds`, `peak_vram`。每个 epoch 再记固定验证图 `bare_A3_size`, `PAVE_size`, `score_time`, `search_steps`。

如果观察到 `alpha` 贴近零、`|delta_prob|` 约 `1e-4`，不能只说“模型收敛”；要结合同信息 MLP/MSG 比较，判断静态信息在 Erdos 目标下是否没有受益梯度。如果 `expected_flips` 偏离目标 5 很大，应优先诊断目标、梯度或输入尺度，而不是再加层数。

---

## 6. P2：必须分开的两种正式质量评价

### 6.1 固定工作量：看 Mamba 是否真的改变采样质量

协议：验证集 100 图 × 3 种子，`K20/T100`，FP32 为主，FP16 为补充；同一模型一次搜索产生相同初始解的 `none`/`pave` 两个后处理版本。

表格必须有：

| Arm | Bare A.3 Size | Δ Bare vs B | +PAVE Size | Δ PAVE vs B | 对 MSG Δ | 对 MLP Δ | feasible | Params | P50/P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 待测 | 0 | 待测 | 0 | — | — | 待测 | 实测 | 实测 |
| MLP | 待测 | 待测 | 待测 | 待测 | — | 0 | 待测 | 实测 | 实测 |
| MSG | 待测 | 待测 | 待测 | 待测 | 0 | 待测 | 待测 | 实测 | 实测 |
| Mamba | 待测 | 待测 | 待测 | 待测 | **主检验** | **主检验** | 待测 | 实测 | 实测 |
| Shuffle-Mamba | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 实测 | 实测 |

建议同时记录：逐图 win/tie/loss、改进节点数分布、最差 5% 图的退化、每种子均值与样本 SD、图级配对 bootstrap 95% CI。统一在图 ID 层重采样，**每个采样图保留其全部 seed 记录**；不要把 300 条“图×seed”当成独立的 300 张图。

### 6.2 固定时间：最关键的真实目标

同一 GPU 独占、每卡同时只允许一个计时进程。交错运行 B 与 v7，不同臂轮换顺序，预热次数一致。`0.5` 和 `1.0` 秒分别测。**图预处理/排序/Mamba 一次性前向应包含在总时间中**。B/v7 皆采用 `--extended-search`，归档逐图完成的 `search_steps`、PAVE 时间、`deadline_overrun`、`incumbent_fallback`。

表格：

| Arm | 0.5s Size | 0.5s steps | 0.5s P95 | 1.0s Size | 1.0s steps | 1.0s P95 | overtime | feasible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B + frozen PAVE | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| MLP + frozen PAVE | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| MSG + frozen PAVE | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Mamba + frozen PAVE | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

**明确限制**：如果固定工作量提升 `+0.05`、但 0.5/1.0 秒预算没赢，则说明方法**不适合作为当前限时部署配置**。不能用“理论 O(N)”“显存只多几 MiB”掩盖实际搜索时间损失。

---

## 7. P3：机制、置换与“是否真的需要 Mamba”的判定

### 7.1 训练后重编号等变/稳定性

对验证图固定 3–5 个同构重编号，采用**节点映射后的共同 Bernoulli 随机张量**，比较：

1. 静态上下文 `g`/`b_add,b_drop` 的逆置换对齐误差；
2. 同状态 `logits` 的对齐差、`|Δp|`、flip decision disagreement；
3. 固定 100 步的 bare A.3 Size，以及冻结 PAVE Size；
4. 原 B 的数值/随机波动下界；单结构序 vs 两结构序平均 vs 四结构序平均；
5. 对最终 Size 的影响是否超过已定义容差，必要时报告“无显著恶化”而非“严格置换等变”。

**数学边界**：双向扫描、按结构分桶再随机 tie-break，只能减轻顺序偏差。有限 `S` 个序的平均仍不保证精确 permutation equivariance；若希望声称严格等变，需要证明排序/随机化分布对重编号的对称性并讨论采样误差，不能用 3 次经验测试代替证明。

### 7.2 Mamba 是否从序列选择中得到特有信息

主检查组合：

- Mamba vs 同 8 维静态输入的 DeepSets/MLP（参数匹配、成本更低）；
- Mamba vs 同 8 维静态输入的一次性 2-hop MSG；
- BFS 排序 Mamba vs **独立训练**的随机排序 Mamba；
- 双向 vs 单向；
- 1 / 2 / 4 views 的置换稳定性与单图前处理耗时；
- 无静态 global inputs / 只局部 inputs（查 global 特征贡献）；
- `alpha=0`、移除 Mamba 静态偏置、仅保留 add 或 drop（查两个动作方向贡献）。

如果仅 `Mamba > B`，但 `Mamba ≈ MSG/MLP`，可报告“额外静态结构先验有帮助”，**不能**把扫描模型当作唯一贡献。如果 `Mamba > MSG/MLP` 只在一个种子正、在固定预算负，也不能主张稳健 SOTA。

### 7.3 弱收益标签不应再次主导主训练

v3 H1 的 7,987 个训练决策状态里只有 286 个非平局，v4/v5 成对续跑标签重复性偏弱；因此不建议 v7 一开始就用单节点反事实收益直接训练 Mamba。可用它们做**机制诊断**，主训练暂时保留 `erdoes`，并以最终 Size 选择。若后续要训练动作收益模型，必须先提高标签可靠性并采用多次 replay/组内置信估计，不再用弱标签堆大模型。

---

## 8. Go / No-go 标准与允许的下一步

### 8.1 建议沿用的硬标准（全部在验证集完成，不能拿作者测试集调参）

| 维度 | 建议 v7 准入 | 说明 |
|---|---|---|
| 正确性 | 100% 可行，零启动等价、无状态/分量泄漏 | 必过 |
| 速度 | FP16/FP32 T100 配对整图比 `≤1.08×` | 训练前/后都要测 |
| 固定 100 步 bare | 三种子均值 `>19.62` 且优于同协议 B | v6 预注册门槛；偏严格，允许如实 No-go |
| 固定 100 步 + PAVE | 每 seed 都超过对应 B，且均值每 seed `>19.69` | 冻结 PAVE |
| 0.5s + PAVE | 三种子均值 `>19.70` 且高于同协议 B | 强对照是持续搜索 B |
| 1.0s + PAVE | 三种子均值 `>19.85` 且高于同协议 B | 同上 |
| 1.0s 搜索步数 | 平均 `≥250` | 沿 v6 验收；必须保留实际步数 |
| Mamba 特有收益 | 相对同信息 MSG 与 MLP 的三种子均为正，**图级配对 95% CI 下界 > 0** | 主科学主张的必要条件 |
| 排序稳健 | 同图重编号后无超过预注册容差的系统性退化，记录 bias/logit/Size 波动 | 不是形式化等变证明 |
| 重复性 | source/teacher/PAVE/feature hashes、状态随机流和完整逐图结果齐全 | 论文可复现前提 |

**不要把 19.62、19.69、19.70、19.85 误写成当前 v7 的已取得结果**；它们只是 v6 方案沿用的高门槛。若 v7 稳定优于 B 却未达到其中某个**绝对值**，应按预注册标准报告“未通过本阶段部署门槛”，可以另外客观报告正向科学证据，不能事后降低门槛然后宣称通过。

### 8.2 三种清晰结局

**Go-A：Mamba 比 B、MSG、MLP 都稳健更好，且两档固定预算均过**  
主文可以写“one-shot static context + Mamba 序列化带来可测、超过廉价等信息控制的增益”。之后只做 1,000 作者测试图一次完整确认、不同 RB 参数/OOD 的泛化测试，以及严谨 ablation，才考虑投稿。

**Go-B：静态结构有益，但 Mamba 没超越 MSG/MLP**  
部署保留便宜的静态 MLP/MSG（如果也通过真实预算）；科研可研究“信息类型、图分布和结构序列化限制”，但不把 Mamba 定成主创新。

**No-go：模型增益不稳、置换伪信号重或在线成本吞噬预算**  
**不继续调一个更大的 Mamba**。主部署恢复 `B sustained search + frozen PAVE H1`；归档 negative evidence、permutation protocol、matched-state trace、latency breakdown。可以把“迭代式组合求解器中一次性全图序列化的收益边界”作为审慎实证研究，但不能提前承诺其发表价值。

---

## 9. 最小可执行的实现顺序与服务器命令模板

以下命令是**v7 新脚本创建后的建议接口**，目前仓库还没有 `train_scripts/v7/`，**不能直接运行并期望成功**。实际脚本需要先按第 3 节实现，且保留旧版本历史脚本可运行。

### Step 1：建分支、锁源代码与环境

```bash
cd /home/hzeng/project/ZH/RLNN_lab
git fetch origin
git switch -c experiment/v7-one-shot-mamba
export PYTHON=/home/hzeng/envs/rlnn/bin/python
export RLNN_SOURCE_SHA="$(git rev-parse HEAD)"
"$PYTHON" -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

Mamba 官方维护的 `state-spaces/mamba` 当前安装细节可能随版本改变。在原环境上先**检查**编译依赖；建议新建隔离环境或保存原环境快照，再按官方 README 安装与固定 wheel/commit。官方当前说明：普通 `pip install mamba-ssm --no-build-isolation` 默认不一定安装 `selective_scan_cuda`；要用 CUDA 扩展须显式 opt-in，并验证 kernel 真正可运行：

```bash
# 示例：须与服务器 torch/CUDA 和编译链兼容；执行前先看官方仓库当前 README。
MAMBA_KEEP_CUDA_BUILD=TRUE "$PYTHON" -m pip install mamba-ssm --no-build-isolation
"$PYTHON" -c 'from mamba_ssm import Mamba; import torch; m=Mamba(d_model=32,d_state=16,d_conv=4,expand=2).cuda(); print(m(torch.randn(1,256,32,device="cuda")).shape)'
```

不能仅验证 `import mamba_ssm` 就假定 RTX 5090 / PyTorch / CUDA 组合能快速执行 selective scan。安装失败时保留日志，**不要为赶进度把 Mamba 偷换为 GRU 后仍命名 Mamba**。

### Step 2：完成静态模块与 P0 preflight

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON" train_scripts/v7/check.py \
  --device cpu --output outputs/v7_context/preflight/cpu.json
CUDA_VISIBLE_DEVICES=0 "$PYTHON" train_scripts/v7/check.py \
  --device cuda --output outputs/v7_context/preflight/cuda.json
CUDA_VISIBLE_DEVICES=0 "$PYTHON" models/profile_v7_context.py \
  --stage preflight --samples 20 --repeats 5 \
  --output outputs/v7_context/preflight/profile
"$PYTHON" models/probe_v7_context.py \
  --split train --samples 100 \
  --output outputs/v7_context/preflight/information_probe
```

必须先审阅输出：未过 P0 则直接 No-go/修复，**不进入下面矩阵**。

### Step 3：主臂 + 控制训练

```bash
# 仅在 P0 明确通过后执行。不同 GPU 上的任务要避免与正式测速混跑。
for ref in 0 1 2; do
  for arm in v7_mamba v7_mlp v7_message v7_shuffle_mamba; do
    REF_SEED="$ref" CUDA_VISIBLE_DEVICES=1 \
      RUN_NAME="train_${arm}_ref${ref}" EPOCHS=3 \
      bash train_scripts/v7/train.sh "$arm"
  done
done
```

这段是**可靠串行模板**，不是最优两卡调度。使用两 RTX 5090 时可以在 GPU0/1 分摊训练，但所有**正式计时**必须每卡独占一个评测进程。每次训练自动保存完整 `latest_training.pt`、`best.pt`、`training.csv`、`run.json`、随机状态、teacher/hash。

### Step 4：固定工作量、两档预算、排序诊断

```bash
for ref in 0 1 2; do
  export REF_SEED="$ref"
  for arm in baseline v7_mlp v7_message v7_mamba v7_shuffle_mamba; do
    CUDA_VISIBLE_DEVICES=0 RUN_NAME="fixed_${arm}_ref${ref}" \
      bash train_scripts/v7/eval.sh "$arm" --cuda-graph
    CUDA_VISIBLE_DEVICES=0 BUDGET=0.5 RUN_NAME="budget05_${arm}_ref${ref}" \
      bash train_scripts/v7/eval_budget.sh "$arm" --cuda-graph
    CUDA_VISIBLE_DEVICES=0 BUDGET=1.0 RUN_NAME="budget10_${arm}_ref${ref}" \
      bash train_scripts/v7/eval_budget.sh "$arm" --cuda-graph
  done
done
"$PYTHON" train_scripts/v7/decide.py \
  --root outputs/v7_context --output outputs/v7_context/analysis/decision.json
```

脚本约定：`baseline` 读取 `outputs/v2/none_seed$REF_SEED/checkpoints`，v7 读取本轮同 seed adapter；三组 PAVE 都读取 `outputs/v3/stats_h1_seed$REF_SEED/checkpoints/best.pt`，禁止训练/复制修改 PAVE；预算入口强制 `--post pave --extended-search --time-budget-s ... --post-reserve-s 0.05`。

### Step 5：只在验证准入后评估作者测试集

```bash
# 仅当 analysis/decision.json 通过，并且所有配置已锁定。
# 不得再用作者 1000 张图选择 Mamba width/order/epoch。
CUDA_VISIBLE_DEVICES=0 REF_SEED=0 SPLIT=test SAMPLES=1000 \
  RUN_NAME=author_test_v7_mamba_ref0 \
  bash train_scripts/v7/eval.sh v7_mamba --cuda-graph
```

完整主结果需三个训练种子与相同作者图、同样的 B/MSG/MLP 强对照；以上只演示一个入口，不等于完成全部作者测试。

---

## 10. 结果文件、数据治理和论文复现规则

每个 run 必须包含：

```text
run.json:
  version, arm, source_sha, branch, utc_start/end,
  backbone_weight_sha256, adapter_weight_sha256,
  PAVE_sha256, train/validation/test_manifest_sha256,
  static_feature_schema, feature_norm_sha256,
  Mamba_library_version, torch_version, CUDA_version,
  CUDA_graph, precision, GPU, search_seed,
  teacher_state_seed, order_seed, order_views,
  model_params, trainable_params,
  mean_size, seed_size_sd, feasible_rate,
  mean_solve_time_s, p50/p95_solve_time_s,
  mean_search_steps, deadline_overruns, fallback_count,
  profile_precompute_ms, profile_rollout_ms,
  profile_decode_ms, profile_pave_ms

per_graph.csv:
  graph_id, reference_seed, model_seed, search_seed,
  order_seed, relabel_seed, base_size, rule_size, final_size,
  solve_time_s, search_time_s, precompute_time_s,
  A3_time_s, PAVE_time_s, search_steps,
  feasible, deadline_overrun, selected_nodes

analysis/: 
  fixed_summary.csv
  budget_summary.csv
  paired_effects.csv
  permutation_summary.csv
  redundant_info_probe.csv
  profile_breakdown.csv
  decision.json
  decision.md
```

**随机性分解**：`backbone_seed`、`adapter_init_seed`、`teacher_state_seed`、`order_seed`、`search_seed`、`relabel_seed` 必须分别记录。FP16 下随机性与 kernel 非确定性还要单独做 native repeat，不得把同一随机 seed 当成严格的逐位复现承诺。

**论文指标**：必须先声明“相同 fixed work”还是“相同 wall-clock”；主任务指标是合法独立集 Mean Size（越大越好），同时报告可行率、P95 latency、实际搜索步数、参数量、显存。仅对没有最优解标签的图，不凭空写 optimality gap 或 approximation ratio。

**实验伦理/科研表述**：任何 negative outcome 都完整保留，不能只展示收益最高的 seed、只算适配器 forward 耗时或将同一组验证图反复调参后的 bootstrap CI 称为独立外部检验。

---

## 11. 最终优先级：现在应该做什么，不应该做什么

**第一优先级**：先把 `V7StaticContextGCN` + `V7Topology.cached_action_bias` + 兼容 `solvers.py`/CUDA Graph 的最小路径打通。**不碰旧 GCN、不碰 PAVE、不碰旧频域模块**。

**第二优先级**：做 20 训练图的 P0 正确性/一次性计时、图重编号与局部信息代理检验；尤其要确认 static Mamba **没有被 capture 进 100 次 replay**，且静态特征没有在 graph batch 之间串扰。

**第三优先级**：仅在 P0 值得继续时，跑 Mamba、同信息 MLP、同信息 MSG、shuffle-Mamba 的三种子 3 epoch；先看 fixed work，再看固定时间预算，最后根据预注册门槛决定是否值得测试集。

**第四优先级**：如果 Mamba 不能稳定优于廉价对照或吞掉 0.5 秒预算，及时 No-go；部署保留 **B 持续搜索 + 冻结 PAVE-Stats H1**。别用第二层 Mamba、更多 token、动态每步 scan 或新增未验证损失来“救”失败。

---

## 12. 相关正式文献与源码（仅提供本轮核验过的来源）

1. **Feng, S. & Yang, Y.** *Regularized Langevin Dynamics for Combinatorial Optimization*, **ICML 2025**, PMLR 267： https://proceedings.mlr.press/v267/feng25r.html ；RLNN 官方代码：https://github.com/Shengyu-Feng/RLD4CO 。本仓库算法基线与作者 A.3 来源。
2. **Gu, A. & Dao, T.** *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, arXiv:2312.00752；源码：https://github.com/state-spaces/mamba 。Mamba/selective scan 的原始技术与安装来源；**不意味着小图稀疏 GCN 一定更快**。
3. **Wang, C., Tsepa, O., Ma, J., & Wang, B.** *Graph-Mamba: Towards Long-Range Graph Sequence Modeling with Selective State Spaces*, arXiv:2402.00789：https://arxiv.org/abs/2402.00789 ；源码：https://github.com/bowang-lab/Graph-Mamba 。图排序和输入依赖节点处理已有先例。
4. **Behrouz, A. & Hashemi, F.** *Graph Mamba: Towards Learning on Graphs with State Space Models*, arXiv:2402.08678：https://arxiv.org/abs/2402.08678 。图 tokenization、ordering、双向 SSM 与局部图编码已有先例。
5. **Xu et al.** *Rethink Efficiency Side of Neural Combinatorial Solver: An Offline and Self-Play Paradigm*（ECO），arXiv:2602.20730：https://arxiv.org/abs/2602.20730 。2026 年已有 Mamba + NCO 的 TSP/CVRP 工作；其自回归/离线偏好学习效率动机**不可直接搬到 RLNN 迭代二值翻转**。该稿件网络上存在不同版本标题，正式引用以最终核实版本为准。
6. **Bamberger et al.** *On Measuring Long-Range Interactions in Graph Neural Networks*, **ICML 2025**：https://proceedings.mlr.press/v267/bamberger25a.html 。提示“需要长期相互作用”的论断应有测量依据，不能仅由 GCN 层数推导。
7. **Chen, D., Schulz, T., & Borgwardt, K.** *Learning Long Range Dependencies on Graphs via Random Walks*, **ICLR 2025**：https://proceedings.iclr.cc/paper_files/paper/2025/hash/f039f5b34b2c2ae27808cf2a01ac3306-Abstract-Conference.html 。序列化/随机游走 + 局部图编码同样有先例。

**参考文献归档**：建议按仓库既有 `reference_methods/<方法>/upstream.json`、`docs/参考文献/bibliography.tsv`、`docs/参考文献/文献索引.md` 规范建立条目，核验作者/版本/出版页/代码许可后再下载论文和源码；不要将 arXiv 预印本误标成已接受的 CCF A 会议成果。**没有完成独立检索覆盖的工作，不可宣称“首次用 Mamba 解 MIS”。**

---

## 附：实现验收清单（给后续 coding agent 的不可跳过条件）

- [ ] `main.py` 添加独立 `--context_mode`；历史 v1–v6 参数和运行保持兼容。
- [ ] `V7StaticContextGCN` 继承原 GCN 的权重键名；热启动加载 v2 `none_seed*`，原骨干全部一致。
- [ ] `alpha=0` 时固定输入 logits 与 B 一致；首步 alpha 梯度、之后 encoder 梯度非零。
- [ ] Mamba **只有每张图一次静态预计算**；每步只有两个动作先验的廉价点式选择。
- [ ] 对含多图的 graph batch 独立 scan；不串图、不混用不同图的归一化统计。
- [ ] 训练**不复用跨 optimizer.step 的带梯度缓存**；推理在 `policy_topology()` 中缓存无梯度 bias。
- [ ] CUDA Graph replay 不含 Mamba kernel；FP16 输出不因零残差提升 dtype。
- [ ] 结构排序使用独立随机流；relabel 实验对 Bernoulli 随机矩阵做节点同步置换。
- [ ] 数据归一化仅用训练集统计，并保存 feature norm hash。
- [ ] Mamba/MLP/MSG 共享完全同样的静态输入与残差公式；尽量实际参数匹配。
- [ ] 三个 teacher B、PAVE H1 各自匹配；PAVE 权重 hash 不变。
- [ ] fixed-work 与 0.5/1.0 秒 wall-clock 两套报告完全分开。
- [ ] 计时涵盖静态特征/排序/scan/GCN/A.3/PAVE/合法性检查，不偷移预处理到计时外。
- [ ] 报三个种子、逐图配对 CI、win/tie/loss、P95、搜索步数、超时与所有负结果。
- [ ] 100 验证图只选择；过门槛后作者 1,000 测试图只做锁定配置的最终评估。
- [ ] 不因 v7 失败修改历史 B/PAVE 基线，No-go 后恢复已验证的限时方案。

**一句话结论**：v7 应当研究“**一次性、置换稳健的 Mamba 静态结构信号，是否能够成为原 RLNN 的低成本、状态非对称翻转先验**”，而不是“给 GCN 再加一个大模型”。只有当它在**同信息廉价对照**和**真实限时**中都胜出，Mamba 才值得成为论文主线。
