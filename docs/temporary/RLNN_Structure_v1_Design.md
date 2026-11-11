# RLNN 结构补偿 v1：设计与实验方案

> 研究点：**在冻结骨干内部补偿消息传递对结构信号与组合增益的压缩**。
> 起点：B（原 RLNN 骨干，`GCN(1, 128, 5, 1)`）＋ 冻结 PAVE-Stats H1。PAVE 不动。
> 前一条线（一次性静态先验 / Mamba）已归档：[归档总结](../已归档的想法和文件/mamba/README.md)。

## 1. 问题在哪

B 的输入只有一个通道——节点的二值状态。度、邻居占用、翻转带来的目标变化，全部要靠 5 层均值归一化聚合在隐空间里重建。这带来两个具体缺陷：

1. **高度数节点被平均掉。** 归一化邻接 `D^-1/2 A D^-1/2` 让高度数节点的邻居贡献被 `1/sqrt(d_i)` 缩放；而 MIS 里高度数节点恰恰冲突最多、翻转影响最大，是最需要精细判断的对象。度数信息在输入层完全缺失，只能靠聚合的间接痕迹恢复。
2. **纯消息传递对组合增益不敏感。** 「这个节点现在翻成 1 能新增多少覆盖」「翻成 0 能释放多少可重选空间」是逐节点的解析量，B 没有任何直接通道拿到它，只能从邻居状态间接推断。

已归档的静态先验走的是**输出端加一个全局方向**，实测无效，且教训明确：瓶颈在输入信息。本点改在**网络内部**注入**状态相关**的结构与增益通道，这两点是与上一版最本质的差别。

## 2. 三个补偿通道

全部零初始化，因此**第 0 步逐位等于 B**（`check.py` 里对四个模式逐个断言 `torch.equal`）。

### 2.1 `degree`：逐层度数残差

```text
φ(deg_i) = [u, u², cos(πk u), sin(πk u)]   k = 1,2,3,  u = log1p(deg_i) / log1p(n_G)
h^(l) ← h^(l) + W_l φ(deg_i),              W_l = 0 初始
```

- 归一化用**每张图自己的** `log1p(n)`，因此单图推理与 8 图合批的编码完全一致（`check.py` 的批不变量断言）。
- 每层一个独立的 `W_l`，让网络自己决定在哪一层注入多少；不是在输出端加一个常数。
- 与「把度数拼进输入」的区别：输入拼接会被第一层聚合重新平均掉，而残差流注入绕过了聚合。

### 2.2 `gate`：度数条件的聚合缩放

```text
m_i ← m_i · (1 + tanh(g_l) · (u_i − mean_G u)),   g_l = 0 初始
```

这条直接对应「平均掉」的机制：它不改加什么，而是改**每个节点保留多少邻居信息**。`g_l` 为零时是恒等；学到正 `g_l` 表示让高度数节点放大自己的聚合结果，负值表示进一步压制。

### 2.3 `gain`：显式组合增益残差

```text
f_i = [x_i, s_i, d_i − s_i, s_i/d_i, (2x_i−1)(β s_i − 1), d_i/250]
ℓ_i ← ℓ_i + tanh(γ) · MLP(f_i),                   γ = 0 初始
```

`s_i` 是当前被选中的邻居数（由 `A @ x` 得到，复用采样器已经维护的 `adjacency_product`，不新增 SpMM）。`(2x−1)(βs−1)` 是解析的局部能量增益，与 `mis_v1.state_features` 的定义一致。这一项是 B 完全没有的通道。

## 3. 为什么不再"粗糙"

| 粗糙版 | 本版 |
| --- | --- |
| 度数残差只是一个可学常数 | 图归一化对数度的 Fourier 编码（8 维），逐层独立注入 |
| 只在输出端加偏置 | 分别作用在残差流（degree）、聚合输出（gate）、翻转 logits（gain）三处 |
| 无法判断是"没学到"还是"没用" | `compensation_effect` 用同检查点清零给出精确的 logit 增量；并记录 `gate_tanh`、`gain_gate_tanh`、`degree_bias_norm` |
| 单一开关 | `--structure_mode degree|gate|gain|all` 四臂独立消融 |

## 4. 实验矩阵

| 臂 | 说明 | 训练参数 |
| --- | --- | ---: |
| `baseline` | 冻结 B，不训练 | 0 |
| `degree` | 度数残差 | 5 × (10 × 128 + 128) = 7,040 |
| `gate` | 聚合缩放 | 5 |
| `gain` | 增益残差 | 6×16+16 + 16+1 + 1 = 130 |
| `all` | 三者同时 | 7,175 |

协议：骨干冻结，教师为冻结 B，`num_tp=50`、K20、Fp32、graph batch 8、state batch 64、3 epoch × 2000 updates、`lr=5e-4`、`lambd=0.5`（与 B 的 Erdős 训练同口径）。评测沿用 `eval_v3`：验证集前 100 图、A.3、冻结 PAVE H1、K20/T100 与 1000、0.5/1.0 秒两档预算，正式计时每卡一个进程。

## 5. 判读规则（先写死）

每个臂看四个数：

| 读数 | 含义 | 判读 |
| --- | --- | --- |
| `comp_delta_rms` | 补偿带来的 logit 增量（同检查点清零） | < 0.01 视为没学到；0.01–0.1 为弱；> 0.1 为实质 |
| `gate_tanh` / `gain_gate_tanh` | 学到的门幅与方向 | 全零 = 未启用；符号决定放大还是压制 |
| `degree_bias_norm` | 逐层注入强度 | 集中在某一层 = 只在那个深度有用 |
| 配对 ΔSize | 图级 bootstrap | 主判据 |

准入门（`summarize.py`）：`fixed`（T100＋PAVE 的 CI 下界 > 0）、`speed`（≤ B 的 1.08×）、`budget`（0.5 与 1.0 秒都 > 0）、`feasible`。四项全过才 `advance`。

三种典型结局与后续动作：

- **`comp_delta_rms` 很小、ΔSize 为零**：补偿没学到东西 → 提高学习率或延长训练，或先做一次「直接注入真实度数/增益、不学习」的上界臂（见 6）。
- **`comp_delta_rms` 大、ΔSize 仍为零**：通道有效但信号无用 → 说明这两类信息对 B 的决策确实没有增量，可归档（与静态先验同构的结论，但这次是状态相关通道）。
- **`gate_tanh` 单调同号且 ΔSize 为正**：度数调制是有效机制 → 进入三种子与更细的门控消融。

## 6. 预留的上界臂

若学习版全部为零，用**不学习**的上界臂把「信息有没有用」和「能不能学」分开（与静态先验 P0 的做法一致）：

- `oracle-degree`：把真实 `u_i` 直接接到 `degree` 通道（`W_l` 固定为单位向量），相当于"完美度数先验"。
- `oracle-gain`：把解析增益 `(2x−1)(βs−1)` 直接加到 logits（等于让网络看见 v1 里 `MISStateGCN` 的那一维），作为增益通道的上界。

这两条不需要训练，评测即可，成本与一次 fixed 评测相同。

## 7. 与文献的关系

检索范围 2024–2026，13 篇 CCF-A ＋ 1 篇 TMLR ＋ 3 篇预印本，PDF 与索引见[文献索引](../参考文献/文献索引.md)的「结构补偿」一节（S01–S17）。对应关系：

| 文献 | 支持的设计决定 |
| --- | --- |
| S01 Degree Bias Origins（NeurIPS 2024）· S02 Networked Inequality（ICML 2024）· S03 High-Degree Augmentation（KDD 2024）· S04 Signed Degree Bias（AAAI 2025） | 度数偏误来自消息传递本身、高度数节点是独立欠服务子群 → 把度数作为**显式**通道（`degree`），而不是等聚合自己恢复 |
| S05 Residual/Normalization Provably（ICLR 2025）· S06 Rewiring Survey（IJCAI 2026）· S07 Riemannian（NeurIPS 2025） | 残差流与层内改造优于拓扑重连 → 三个补偿都放在层内，`degree` 走残差流 |
| S08 Oversquashing as Dynamics（AAAI 2025）· S09 Curriculum Residual Norm（AAAI 2024） | 邻居信号被收缩 → `gate` 按度数调制"保留多少邻居信息"，且门幅有界 |
| S10 GATE（ICML 2024）· S11 Polynormer（ICLR 2024）· S12 Sketch-Augmented（NeurIPS 2025）· S13 Feature Transportation（AAAI 2024） | 门控与非消息传递特征通道确有增量 → `gate` 的乘法形态、`gain` 的显式特征残差与零初始化 |
| S14 DiffQuadratic MIS（ICML 2025）· S15 MARCO（IJCAI 2024）· S16 LoRe（ICML 2026） | 组合优化的局部增益与状态回灌 → `gain` 的特征选型（选中邻居数、未覆盖邻居数、解析增益） |
| S17 Unrealized Expectations（TMLR 2026） | 学习式 MIS 的诚实对照 → 定量表保留 KaMIS 与 B 的绝对参照，不用单一均值宣称收益 |

可参考代码快照（作者仓库，pin 到 commit）：Polynormer（纯特征全局通道）、Sketch-Augmented（层内 sketch 特征注入）、DiffQuadratic-MIS（MIS 可微二次优化与基准）。见 [reference_methods](../../reference_methods/README.md)。

## 8. 运行

见 [train_scripts/structure/README.md](../../train_scripts/structure/README.md)。输出写入 `outputs/structure_v1/`，判决写入 `outputs/structure_v1/analysis/`。
