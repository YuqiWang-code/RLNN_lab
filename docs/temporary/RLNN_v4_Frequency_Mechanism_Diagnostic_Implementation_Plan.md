# RLNN 图频域模块：下一轮机制诊断实验与代码改造实施方案

> **项目**：[`YuqiWang-code/RLNN_lab`](https://github.com/YuqiWang-code/RLNN_lab)  
> **核查基准提交**：[`013cbcbf1a55e77548abf07d4cdf656e99c097a3`](https://github.com/YuqiWang-code/RLNN_lab/tree/013cbcbf1a55e77548abf07d4cdf656e99c097a3)，2026-11-09  
> **研究任务**：RB200–300 最大独立集（MIS），只分析图高低频模块；**PAVE-Stats H1 权重、代码与部署协议保持不变**  
> **文件性质**：基于已提交代码和日志的**下一轮可执行实验设计**，不是已运行的新结果；所有“新增脚本／建议改动”均未写入 GitHub。

---

## 0. 先给出实施结论

**下一步不是训练 v5，而是用 v4 已训练的 12 个权重完成四类机制实验，回答四个不同的问题：**

1. **频带保真与信息去向**：高通输入经过哪一层后变成低频占主导？是偏置／非线性／LayerNorm／跨通道混合，还是度数加权和常数分量的统计口径造成的表象？
2. **分支因果作用（算法干预，而非因果识别声明）**：固定同一个模型、同一个状态与同一条随机流，关闭 low/high、固定 gate、改变残差系数，哪种改变能提高即时行为质量、全轨迹 Size？
3. **动作收益与任务相关性**：固定图、节点、状态、续跑随机数，使用更多重复次数测“翻转节点”和“不翻转”的未来解码收益。低高频分支是否对有利动作提供了**超出普通消息编码**的预测信息？
4. **完整计算开销**：频率过滤、双分支 MLP、门控、GCN、A.3、PAVE 分别占多少时间？性能下降有多少来自同样 0.5 秒内搜索步数减少？

**判断顺序**：先复核历史证据与正确性 → 频率去向 → 分支干预 → 成对收益 → 耗时复测 → 出具 Go / No-go 决策。只有发现可靠的任务相关正向信号，才讨论新结构及训练。

### 0.1 已有事实（不是本轮新结论）

上一轮仓库已经写明：v4-A 在 **100 张验证图 × 3 个种子**下，FP32、固定 K=20/T=100、无 PAVE 的 Mean Size 为 **19.5300 ± 0.0265**，低于同信息／同参数普通消息模型 **19.5467 ± 0.0058**。FP16 固定工作量下分别是 **19.4667** 和 **19.5100**；0.5 秒内加冻结 PAVE 分别为 **19.4800** 与 **19.5200**，而原骨干＋冻结 PAVE 为 **19.6233**（允许原骨干继续搜索则为 **19.6500**）。

对已训练 F4 的直接输出干预：关闭 low 后 Size 为 **19.5400**，关闭 high 后 **19.5500**，完整 F4 为 **19.4667**（FP16 固定工作量、三个种子）。输入 `high_input=[B0,B1]` 的较高频能量约 **90.89%**，而高频 MLP 输出 `u_high` 的低频能量约 **97.21%**。成对单节点翻转收益排序正确率 F4 约 **48.84%**，普通消息约 **48.88%**，二者没有表现出有效优势。

**重要边界**：这些数字属于 100 张验证图的短期结构筛选。v4 共训练 **3 epoch／6000 次更新**，共享冻结教师状态；不应声称已在 1000 张正式测试图上证实 v4 失败，也不能推导“图频率信息对所有 MIS 搜索无价值”。

历史材料入口：

- [v4 已发表在仓库内的结果分析](https://github.com/YuqiWang-code/RLNN_lab/blob/013cbcbf1a55e77548abf07d4cdf656e99c097a3/docs/temporary/RLNN_v4_Frequency_Results_Analysis.md)
- [v4 汇总记录](https://github.com/YuqiWang-code/RLNN_lab/blob/013cbcbf1a55e77548abf07d4cdf656e99c097a3/outputs/v4_frequency/summary.txt)
- [阶段决策 `stop_current_adapter`](https://github.com/YuqiWang-code/RLNN_lab/blob/013cbcbf1a55e77548abf07d4cdf656e99c097a3/outputs/v4_frequency/decision.json)
- [v3 PAVE 已验证方案](https://github.com/YuqiWang-code/RLNN_lab/blob/013cbcbf1a55e77548abf07d4cdf656e99c097a3/docs/temporary/RLNN_v3_Results_Analysis.md)

### 0.2 本轮绝对不改什么

- **不改** `models/mis_v3_search.py` 的合法平台搜索逻辑、候选集合、H=1、搜索与修复次数。
- **不改** `models/mis_v3_value.py` 及 `outputs/v3/stats_h1_seed{0,1,2}/checkpoints/best.pt`。
- **不改** RLNN 的 MIS 能量、`erdoes` 损失、K=20、T=100、`β=1.02`、`d=5`、A.3 解码。
- **不在**正式 1000 张测试图上挑结构；本轮只用训练图、验证图及这些图上构造的固定轨迹状态。
- **不因“高通输出低频比例高”就直接强制正交、删掉低通或引入新辅助损失**；先弄清该现象是否有害。

---

## 1. 当前代码到底做了什么：文件和运算关系

### 1.1 `models/` 关键文件分工

| 文件 | 当前已存在的功能 | 下一步具体操作 |
|---|---|---|
| [`models/mis_v4_frequency.py`](https://github.com/YuqiWang-code/RLNN_lab/blob/013cbcbf1a55e77548abf07d4cdf656e99c097a3/models/mis_v4_frequency.py) | `v4_topology`、`state_bands`、`V4FrequencyAdapter.inspect`、`V4FrequencyGCN` | **保留模型与参数名**；补无参数的中间层跟踪函数、同状态残差替换接口 |
| [`models/diagnose_v4_frequency.py`](https://github.com/YuqiWang-code/RLNN_lab/blob/013cbcbf1a55e77548abf07d4cdf656e99c097a3/models/diagnose_v4_frequency.py) | 按 t=0/10/25/50/75/100 采样；输出 bands/spectra/gates/energy/conflicts | 扩展中间层谱、驻留模式去除、分组行为与诊断置信区间 |
| [`models/probe_v4_utility.py`](https://github.com/YuqiWang-code/RLNN_lab/blob/013cbcbf1a55e77548abf07d4cdf656e99c097a3/models/probe_v4_utility.py) | 单节点成对续跑、冻结 RLNN 教师、当前 R=2 次重复、收益排序 | 将 R 改为 CLI 可选，分层采样节点，保存逐重复的收益，增加标签可靠性分析 |
| [`models/eval_v4_frequency.py`](https://github.com/YuqiWang-code/RLNN_lab/blob/013cbcbf1a55e77548abf07d4cdf656e99c097a3/models/eval_v4_frequency.py) | 正式输出、冻结 PAVE 约束、`--disable-low/high/gate` | **保持原评测口径**；诊断开关不混入正式主表 |
| [`models/solvers.py`](https://github.com/YuqiWang-code/RLNN_lab/blob/013cbcbf1a55e77548abf07d4cdf656e99c097a3/models/solvers.py) | `RLNN.policy_topology`、`flip_logits`、`sample_states`，训练与损失 | 第一阶段尽量**零修改**；后续新增独立 replay 驱动，不破坏默认搜索 |
| [`models/eval_v2.py`](https://github.com/YuqiWang-code/RLNN_lab/blob/013cbcbf1a55e77548abf07d4cdf656e99c097a3/models/eval_v2.py) | `dataset_at`、`load_policy`、`search`、A.3、同图计时 | 直接复用图加载和原算法求解，不修改默认输出 |
| `models/models.py` / `models/problems/mis.py` | 残差 GCN 与 MIS 能量、梯度翻转解码 | 禁止改动，以免诊断与原研究目标不再一致 |
| **建议新增** `models/frequency_diag_utils.py` | 当前没有 | 将谱投影、阶段跟踪、节点分组、聚合统计集中在纯函数中 |
| **建议新增** `models/diagnose_v4_branch_replay.py` | 当前没有 | 固定初始状态＋预生成随机流，做分支扰动的成对完整搜索 |
| **建议新增** `models/profile_v4_frequency.py` | 当前没有 | 独占 GPU 的步骤级耗时剖析，避免诊断计算污染正式计时 |
| **建议新增** `tools/summarize_frequency_followup.py` | 当前没有 | 跨种子、跨图、逐图配对汇总以及最终实验门槛决策 |

### 1.2 v4 频率输入与结构

原始张量 `x ∈ {0,1}^{N×K}`，其中 **N 是节点，K=20 是独立粒子，不是图像像素的第二个空间维**。图传播只在节点维进行。

令 `c=Ax` 为每个节点的已选邻居数（这里 A 是二值邻接，代码中的 MIS 能量邻接边权为 0.5，因此必须区分）。代码使用：

\[
Z=[x,\log(1+c)]\in\mathbb{R}^{N\times K\times 2},\quad
P=\tfrac12(I+D^{-1}A),
\]

\[
L_2=P^2Z,\quad B_0=Z-PZ,\quad B_1=PZ-P^2Z,
\]

\[
\boxed{Z=L_2+B_0+B_1}.
\]

代码中 low 是 `2→16→128`，high 是 `4→16→128`，两支 `Linear→SiLU→Linear→LayerNorm`，分别乘独立 sigmoid 门控及可学习残差系数后相加到原始 GCN embedding。随后仍通过**原五层 GCN**预测各节点翻转 logits。

当前核心函数：

```text
models/mis_v4_frequency.py
  v4_topology(edge_index, adj, nodes)
  state_bands(x, topology, beta)
  V4FrequencyAdapter.inspect(x, adj, topology)
      -> z, pz, ppz, L2, B0, B1, low_input, high_input,
         gate_input, gates, u_low, u_high, delta_low, delta_high
  V4FrequencyGCN.forward(x, edge_index, adj, topology=None,
                         disable_adapter=False)
```

**这意味着第一轮完全不必重写网络**：当前 `inspect` 已经暴露绝大部分必要中间量，只需补更细的编码阶段跟踪、受控 replay 和严谨的统计口径。

### 1.3 先写出需要证伪的假设

| 编号 | 待检验假设 | 什么结果会否定它 |
|---|---|---|
| H-F1 | 高频编码有用，只是编码器使高频信息受损 | 去除偏置／调整非线性后，动作收益与最终 Size 仍无改善 |
| H-F2 | 两路残差产生相互抵消或错误竞争 | 同状态隔离两路后，不存在一致的负交互；重训练消融也无提升 |
| H-F3 | 频带对少数关键动作有效，但被总体平均指标淹没 | 在冲突节点、唯一阻挡节点、易释放节点等子集上也无可靠优势 |
| H-F4 | 表示质量接近，但额外 SpMM 造成同时间搜索不足 | 同 K/T 下已明显更差，成本优化不能弥补质量差；或优化后仍无同预算收益 |
| H-F5 | 固定教师轨迹与学生最终状态分布错位 | 在共同状态和学生自生成状态上都没有额外价值，继续 on-policy 无依据 |

注意这些是假设，不是已经证明的原因。实验要允许它们被否定。

---

## 2. 实验总体设计：数据、模型、预算与目录

### 2.1 模型清单

至少加载相同种子下的五个已存在权重（不用重训）：

| 记号 | 仓库中的训练目录 | 作用 |
|---|---|---|
| B | `outputs/v2/none_seed{seed}/checkpoints/best.pt` | 原 RLNN 教师、共同轨迹来源、无适配器参考 |
| B+FT | `outputs/v4_frequency/none_seed{seed}/checkpoints/best.pt` | 相同短期训练预算的纯骨干微调 |
| MLP | `outputs/v4_frequency/v4a_mlp_seed{seed}/checkpoints/best.pt` | 同参数状态适配器 |
| MSG | `outputs/v4_frequency/v4a_message_seed{seed}/checkpoints/best.pt` | 与频域输入线性等价的普通消息对照 |
| F4 | `outputs/v4_frequency/v4a_seed{seed}/checkpoints/best.pt` | 需要诊断的多尺度低／高频适配器 |

所有 `{seed}` 用 0、1、2，各模型的训练种子与 PAVE 头种子配对。先检查路径存在、hash 值、模型加载、eval 模式、`requires_grad_(False)`；不要默默回退到其他权重。

冻结 PAVE（**只在最终解质量/完整计时实验中使用**）：

```text
outputs/v3/stats_h1_seed{seed}/checkpoints/best.pt
```

### 2.2 数据分层

- **层 A：实现自检**——7 节点玩具图、孤立点、两张图块对角合批、粒子维置换，不作为效果证据。
- **层 B：探索性机制诊断**——训练集上预先固定的 20–50 图、每图 t=0/10/25/50/75/100；包括模型内部层的诊断、候选动作标签定义和性能剖析。
- **层 C：复核机制诊断**——100 张原验证图、3 个模型种子，用于复核层 B 的主要现象；**这 100 张图已用于 v4 选权重，不能把反复探索后的表现当成独立测试证明**。
- **层 D：最终保留集**——官方 `idx_0…idx_999` 的 1000 张测试图，只能在有明确、预先锁定的下一版设计后运行一次正式评估。现阶段不碰。

如果训练集中另划机制开发子集，应按**图 ID/生成 seed**固定并输出 manifest，避免与验证/测试混合。只从已有缓存读取图，不重新生成声称同一分布却不同的文件。

### 2.3 搜索和评价恒定配置

```yaml
problem: mis
beta: 1.02
num_k: 20
num_t: 100
num_d: 5
num_l: 5
num_h: 128
mixed_precision:
  diagnostic_spectrum: fp32
  deploy_evaluation: fp16
search_seed: 1017  # 每图再加 graph_index
postprocess: none  # 独立诊断时
pave_when_required:
  implementation: v3_stats_h1_frozen
  horizon: 1
  outer_rounds: 8
  initial_exchange_rounds: 3
  candidate_cap: 32
  improving_cap: 48
  repair_rounds: 4
fixed_time_budget_s: 0.5
pave_reserve_s: 0.05
```

**分开统计两个实验宇宙**：

- **固定工作量**：K20/T100，相同阶段状态可复用；报告 Size、单图总求解时间、每阶段额外时间。
- **固定时间**：0.5 s/图，使用原 `eval_v4_frequency`；允许各方法实际搜索步数不同，报告 Mean Size、执行步数、平均和 p95 时间、超时数量。不能将两个宇宙的 Size 混算。

### 2.4 推荐输出目录：全新命名，绝不覆盖 v4

```text
outputs/v4_frequency/followup/
├── manifest.json                  # SHA、硬件、种子、数据子集、权重hash
├── p0_checks/                     # 运行正确性、数值比对、规模测试
├── p1_spectrum/
│   └── {model}_seed{n}/
│       ├── stage_spectra.csv      # 各层谱能量占比与绝对能量
│       ├── stationary_removed.csv # 去驻留分量后比例
│       ├── gates_and_deltas.csv
│       └── run.json
├── p2_branch/
│   └── seed{n}/
│       ├── same_state.csv         # 同状态 logits/probabilities / expected flips
│       ├── replay_per_graph.csv   # 同一 x0、同一 U 下的完整轨迹
│       └── run.json
├── p3_utility/
│   └── seed{n}/
│       ├── state.csv              # 图、状态、t、候选类别分布
│       ├── action_replicate.csv   # 每个候选、每个重复的真实增益
│       ├── action_aggregate.csv   # 重复均值、标准误、收益符号稳定性
│       ├── ranking.csv           # 排序、top1、regret、子集分组
│       └── run.json
├── p4_profile/
│   └── seed{n}/
│       ├── components.csv         # 图滤波/编码/门控/GCN/解码/PAVE耗时
│       ├── end_to_end.csv         # 正式独占计时
│       └── run.json
├── analysis/
│   ├── evidence_matrix.md         # 每个假设支持/不支持/未判定
│   ├── paired_graph_results.csv
│   ├── summary.json
│   └── decision.json
└── README.md
```

### 2.5 研究流程图

```text
 P0 复核仓库提交、权重、数据和单元测试
  │
  ├─失败 → 先修实现，暂停所有结论
  │
  ▼
 P1 固定状态频谱与分支编码阶段跟踪
  │   找到真正改变频谱的阶段 + 判断是否只由驻留/常数项主导
  ▼
 P2 同状态部件干预 + 共享随机数全轨迹 replay
  │   判断分支是“有用但被干扰”，还是无有效增益
  ▼
 P3 分层采样节点 + 多重复成对收益
  │   判断分频是否在关键节点上提供额外动作排序价值
  ▼
 P4 独占 GPU 计时与算子耗时分解
  │   判断即便有收益，是否被时间成本抵消
  ▼
 证据矩阵 + 明确下一轮结构变更；PAVE 保持冻结
```

---

## 3. P0：实施前的正确性与可复现性检查

### 3.1 先复用已有的自检文件

仓库已经有 `train_scripts/v4_frequency/check_adapter.py`，它验证了：

1. CSR 邻接中单位边权与能量计算的 0.5 边权语义一致；
2. `L2+B0+B1 == Z` 的数值重构；
3. 孤立点惰性游走恒等；
4. 不加载适配器残差时与原 GCN 完全一致；
5. 节点重编号等变、粒子重排等变；
6. 多图块对角合批等价与梯度相容；
7. fp16 前向有限、参数量一致：原 RLNN 165,505，新增适配器 5,008。

在项目根目录执行现有命令：

```bash
export PYTHON=/home/hzeng/envs/rlnn/bin/python
export CUDA_VISIBLE_DEVICES=0
$PYTHON train_scripts/v4_frequency/check_adapter.py \
  --device cpu --output outputs/v4_frequency/followup/p0_checks/check_cpu.json
$PYTHON train_scripts/v4_frequency/check_adapter.py \
  --device cuda --output outputs/v4_frequency/followup/p0_checks/check_cuda.json
```

**执行前先创建对应目录**：

```bash
mkdir -p outputs/v4_frequency/followup/p0_checks
```

脚本不是新的，命令应可复用。若环境/路径不同，改 `PYTHON`、`CUDA_VISIBLE_DEVICES`，不改算法。

### 3.2 新增的四项一致性测试

在现有 `check_adapter.py` 追加或新建 `tests/test_v4_followup.py`：

**检查 A：阶段跟踪一致。** 对同样的 `high_input`，逐层手动调用 `adapter.high[0..3]` 的输出必须与 `adapter.high(high_input)` 逐元素相同，误差建议 FP32 `atol=1e-6`。

**检查 B：非侵入性。** 开启诊断（仅计算、记日志）时，同一个 `x` 的模型 logits 必须与不诊断时一致；训练检查点键名、权重 hash 不变。

**检查 C：相同图同状态不同粒子不能互相影响。** 先对各粒子单独前向，再把粒子拼起来，结果一致。`x[N,K]` 的 K 不能被当作节点维度。

**检查 D：固定随机流。** 同一模型以同一个 `x0` 和预生成的 `U[step,node,particle]` replay 两次，输出逐步相同；不同方法的共同随机数数组字节一致。

**失败处理**：上述检查未通过，不开展 P1–P4，不能利用后续统计解释模型机制。

---

## 4. P1：频带是否在编码阶段失去作用？怎样修改 `models/`？

### 4.1 本组真正要检验什么

现有 `diagnose_v4_frequency.py` 记录了 **`high_input` 与完整 `u_high`**，缺少精确定位：

- `high_input` → 第一层 `Linear(4,16)`；
- → `SiLU`；
- → 第二层 `Linear(16,128)`；
- → `LayerNorm(128)`；
- → 乘 gate 及 `alpha_high`；
- → 与原始 embedding 相加；
- → 经过 5 层 GCN 的最终翻转 logits。

必须加上这些中间状态的谱统计，同时做去驻留分量后的对照。**目标不是强迫 high 分支永远是高频，而是判断何处产生了低频主导，以及这种改变是否关联到动作能力下降。**

### 4.2 最低风险改动：新增纯函数，不动已有权重

新建 `models/frequency_diag_utils.py`，加入以下可直接移植的纯函数：

```python
import torch

@torch.no_grad()
def trace_sequential(block: torch.nn.Sequential, x: torch.Tensor):
    """按原有 Sequential 顺序跟踪。不会注册参数或改变 state_dict。"""
    out = x
    stages = {"input": out.detach()}
    for index, layer in enumerate(block):
        out = layer(out)
        stages[f"stage{index}_{type(layer).__name__}"] = out.detach()
    return stages

@torch.no_grad()
def spectral_energy_extended(signal, basis, eps=1e-7):
    """与已有 spectral_basis 完全相同的 L_sym 基，额外排除零特征值空间。"""
    eigenvalues, eigenvectors, sqrt_degree = basis
    s = signal.detach().to("cpu", dtype=torch.float64).flatten(1)
    weighted = sqrt_degree[:, None].to(torch.float64) * s
    coeff = eigenvectors.T @ weighted
    e = coeff.square().sum(dim=1)
    zero = eigenvalues.abs() <= eps
    nonzero = ~zero
    e_centered = e * nonzero
    result = {
        "total_energy": e.sum().item(),
        "stationary_fraction": (e[zero].sum() / e.sum().clamp_min(1e-30)).item(),
        "nonstationary_energy": e_centered.sum().item(),
    }
    for lo, hi, name in [(0.0, 0.5, "low"), (0.5, 1.0, "mid"),
                         (1.0, 1.5, "higher"), (1.5, 2.000001, "extreme")]:
        mask = (eigenvalues >= lo) & (eigenvalues < hi)
        result[f"raw_{name}_fraction"] = (
            e[mask].sum() / e.sum().clamp_min(1e-30)
        ).item()
        result[f"centered_{name}_fraction"] = (
            e_centered[mask].sum() / e_centered.sum().clamp_min(1e-30)
        ).item() if e_centered.sum() > 1e-30 else float("nan")
    return result
```

**说明**：

- `basis` 直接使用当前 `diagnose_v4_frequency.py::spectral_basis` 返回的 `(eigenvalues,eigenvectors,sqrt_degree)`；切勿在这里自己换成另外一种拉普拉斯、混用图的节点排列。
- 现有 `spectral_basis` 对孤立节点的处理以该实现为准。`eps` 是数值阈值，必须先在玩具图与不同精度上检查特征值误差。零空间可能不止一个维度，不能简单删第 0 个特征向量。
- `centered_*` 只是**诊断量**，不是把模型输入做去均值，也不是建议修改训练网络。不同层通道数不同，比较能量比例同时必须记录**绝对能量**。
- 如果输出几乎只有驻留分量，分母接近零，记为缺失值和 `valid_centered=false`；不要把 NaN 转成 0 再做均值。
- 现有 RB 图最大实测特征值约 1.396，因此 `[1.5,2]` 频带为空是正常的；必须保存每图真实谱支持，不能拿空频带进行模型排名。

### 4.3 在 `diagnose_v4_frequency.py` 中接入阶段跟踪

当前 `observe()` 已有：

```python
with solver.accelerator.autocast():
    details = adapter.inspect(x, adj, ft)
    ...
```

**改法**：保留上面计算，在 `adapter is not None` 分支中调用：

```python
from frequency_diag_utils import trace_sequential, spectral_energy_extended

# 位置：observe() 内 details 已产生、x/ft/basis 已确定之后。
# 仅用于诊断，不改变模型forward；保持与原有autocast上下文一致。
with solver.accelerator.autocast():
    low_trace = trace_sequential(adapter.low, details["low_input"])
    high_trace = trace_sequential(adapter.high, details["high_input"])

signals.update({f"low_{name}": value for name, value in low_trace.items()})
signals.update({f"high_{name}": value for name, value in high_trace.items()})

# 对每个阶段，不仅保留原谱统计，还另算去驻留成分的谱占比。
if basis is not None:
    for name, value in signals.items():
        record = spectral_energy_extended(value, basis)
        stage_rows.append({
            "graph_id": graph_id, "step": step, "signal": name,
            "seed": backbone_seed, "particles": particles, **record,
        })
```

上面是**函数插入示例**：必须在现有 `diagnose()` 内先定义 `stage_rows=[]`，结束时保存到 `stage_spectra.csv`，且注意 `signals` 的创建位置（当前文件里是在 `band_row` 之前）。请不要把示例直接粘在未定义变量的作用域。

另外校验：

```python
# FP32诊断时应非常接近；FP16/AMP 路径另单独给更宽阈值。
assert torch.allclose(high_trace["stage3_LayerNorm"],
                      details["u_high"], atol=1e-5, rtol=1e-5)
assert torch.allclose(low_trace["stage3_LayerNorm"],
                      details["u_low"], atol=1e-5, rtol=1e-5)
```

### 4.4 必须增加“驻留分量—偏置—非线性”三类对照

**P1-A：不改模型权重，只看阶段变换。**

完整记录 `input/Linear1/SiLU/Linear2/LayerNorm/gate×alpha/最终 logits`。如果高频的低频占比在 `Linear1` 就跃迁，优先查看该层偏置与输入均值；如果在 `SiLU` 或 `LayerNorm` 后跃迁，则分别关注非线性与通道归一化。

**P1-B：输入不变，仅在诊断函数中做偏置置零模拟。**

不要直接修改 `module.bias.data`；用 `torch.nn.functional.linear(x,layer.weight,None)` 计算无偏置的诊断分支，再将输出按现有模块顺序走完。对第一层、第二层分别运行“仅去一层偏置”的对照，记录每阶段绝对谱能量、去驻留后的比例和输出 logits 变化。

这个操作属于**固定权重推理干预**，不能报告为经过重训练的无偏置网络性能。由于 LayerNorm 会重新标准化，去掉某层偏置可能同时改变后续激活分布，务必逐层跟踪。

**P1-C：只对谱统计移除零频投影。**

对 `u_high` 在图谱上只扣除零特征值子空间的投影，然后重新计算 `raw/centered`；不把扣除后的值送回模型。若 raw 的 97% 低频在去驻留后明显下降，说明原指标受共同偏置／驻留成分影响较大，**不能直接把 97% 解释成高频信息被破坏**。

### 4.5 进一步量化“信息有没有真的影响决策”

只看能量比例不够。对同一状态：

- 计算 `delta_low`、`delta_high` 与主干 `hidden` 的 `RMS`、`norm ratio`；
- 计算启用／关闭 high 后的 `Δlogit` 和 `Δp`，并按节点类别分组；
- 计算输入高频各分量 `B0/B1` 与 `Δlogit` 的相关性（**描述性相关**，不解释为独立因果作用）；
- 计算 `||Δlogit||` 在 λ∈[0,0.5)、[0.5,1)、[1,1.5) 的能量、去驻留比例；
- 单独统计 `p<0.01`、`p>0.99` 的节点占比：logits 明显变化但 sigmoid 几乎不变，可能源于饱和；
- 汇总每个图的指标，再对图求均值；**不能将所有节点、所有粒子视为独立样本**。

可增加一次只针对最后一层的表示诊断：从输入到每个 GCN Block 的隐藏状态做谱统计，确定主干是否进一步平滑。但注意五层 GCN 的激活/残差都可能混合频率，不应要求中间高频严格保持纯净。

### 4.6 P1 推荐规模与验收结果

| 阶段 | 模型 | 图数 | 粒子 | 时刻 | 是否训练 |
|---|---|---:|---:|---|---|
| 冒烟 | F4 seed0 | 3 | 4 | 0、25、100 | 否 |
| 探索 | F4、MSG、MLP seed0 | 20 | 4 | 0、10、25、50、75、100 | 否 |
| 复核 | F4、MSG、MLP，seed0/1/2 | 50–100 | 4 | 同上 | 否 |

应交付两张图和两张表：**阶段频谱堆叠图**（原始、去驻留后分开）、**残差对 logits 的逐频段贡献图**、按种子分组的**频谱数表**和**行为变化数表**。

**P1 不能单独决定上新结构。** 只要没有直接动作收益或最终 Size 的正向证据，即使“高频比例恢复”也不等于创新成功。

---
## 5. P2：同状态分支干预与共享随机流 replay

### 5.1 这组实验为什么必须重做得更严谨？

v4 已经有关闭 low/high/gate 的结果，但该实验的输出是**每个模型自己采样出的不同搜索状态**：关闭一个分支不只改变单步 logits，也会让后续随机轨迹走向不同区域。它给出了重要的“完整算法反应”，却无法直接解释是哪些状态上的哪种动作改变了最终结果。

本轮必须并行记录**两类互补证据**：

- **同状态、一步干预**：固定 `G,x,θ`，比较同一已训练 F4 的 `full / no-low / no-high / gate=0.5` 的 logits、概率和采样动作。**状态不许改变**，所以能定位分支对某次决策的影响。
- **同初始化、共享随机流的完整 replay**：固定 `G,x0,U[0:T]`，允许状态在干预后自然分叉，记录最终历史最低能量解的 A.3 Size。它能测量该**算法干预在指定共同噪声下**的完整后果，但两条轨迹不再逐步相同。

不能把两者混成同一种“因果效应”；只把它们称为**受控算法干预／配对随机实验**。

### 5.2 需要新增的文件和接口

新增 `models/diagnose_v4_branch_replay.py`（不要修改现有训练／评测入口）：

```text
load_group(seed, graph_id):
    B, B+FT, MLP, MSG, F4 = load frozen models
    G = same graph
    x0 = same [N,K] binary tensor
    U = same [T,N,K] uniform random tensors

analyze_same_state(G, x_teacher_t, model_F4):
    logits_full, logits_no_low, logits_no_high, logits_gate_half
    p_full, p_no_low, p_no_high, p_gate_half
    gate, residual norms, conditional summaries

replay(G, x0, U, model, intervention):
    for t in [0..99]:
        logits = model(current_state; intervention)
        flip = U[t] < sigmoid(logits)
        update current_state
        update per-trajectory best-energy archive
    return A3-decode best states and max-over-K Size

write per_graph.csv / per_state.csv / run.json
```

### 5.3 先扩展 `mis_v4_frequency.py` 的非侵入式观察入口

**不要新增参数，不改变旧模型 checkpoint。** 当前已有 `adapter.low_override / high_override / gate_override`，可直接用于序列测试。但全局成员在多个任务共享时容易残留，建议只在独立诊断模型副本上修改，结束后恢复：

```python
from contextlib import contextmanager

@contextmanager
def adapter_override(adapter, *, low=None, high=None, gate=None):
    old = (adapter.low_override, adapter.high_override,
           adapter.gate_override)
    try:
        adapter.low_override = low     # None=原状；False=关闭
        adapter.high_override = high
        adapter.gate_override = gate   # None=原状；0.5=固定门控
        yield
    finally:
        (adapter.low_override, adapter.high_override,
         adapter.gate_override) = old
```

**用法**：

```python
with adapter_override(model.adapter, low=False):
    logits_no_low = model(x, edge_index, adj, topology)
```

同状态检查的输入 `x` 应由**固定教师 RLNN**的状态采样得到，并存储图 ID、搜索 t、粒子 ID。每个模型的分支干预都用同样的状态集合；不可在比对时重新调用 `sample_states()` 让输入发生改变。

### 5.4 完整 replay 的核心循环（参考可移植代码）

建议在新文件中按下面逻辑实现。这里是**核心算法骨架**，图加载和模型构造请复用 `eval_v2.py` 的现有函数；不宣称这段片段无需封装即可独立运行。

```python
import torch
from mis_v1 import adjacency

@torch.no_grad()
def replay_one_graph(solver, graph, x0, uniforms):
    """uniforms: [T,N,K]，严格一次性预生成；不调用额外随机数。"""
    adj = adjacency(graph)
    topo = solver.policy_topology(graph, adj)
    x = x0.clone()
    best = x.clone()
    best_e, _ = solver.problem.energy_func(adj, graph.b, x)
    phase_rows = []

    for t in range(uniforms.shape[0]):
        logits = solver.flip_logits(x, graph, adj, topo)
        prob = logits.sigmoid()
        flips = uniforms[t] < prob
        x = torch.where(flips, 1 - x, x)
        energy, _ = solver.problem.energy_func(adj, graph.b, x)
        better = energy < best_e
        best = torch.where(better.unsqueeze(0), x, best)
        best_e = torch.where(better, energy, best_e)
        if t in (0, 9, 24, 49, 74, 99):
            phase_rows.append({
                "step": t + 1,
                "best_energy_mean": float(best_e.float().mean()),
                "expected_flips_mean": float(prob.sum(dim=0).float().mean()),
                "actual_flips_mean": float(flips.sum(dim=0).float().mean()),
            })

    solutions = solver.problem.decode_gradient_flip(
        best, adj, graph.b, return_solutions=True)
    sizes = solutions.sum(dim=0)
    return int(sizes.max().item()), phase_rows
```

**调用时**：每张图仅生成一次 `x0` 和 `uniforms`：

```python
rng = torch.Generator(device=device).manual_seed(1017 + graph_index)
x0 = torch.randint(0, 2, (graph.num_nodes, 20),
                   device=device, dtype=graph.b.dtype, generator=rng)
u = torch.rand((100, graph.num_nodes, 20),
               device=device, dtype=graph.b.dtype, generator=rng)
```

两个注意事项：

1. `x0/u` 的 dtype 必须与模型推理精度一致，不能让 F4 用 FP32 均匀数、Baseline 用 FP16 均匀数；若严格共享随机数，建议先在固定 FP32 下测机制，再单独做 FP16 版本。
2. 这个 `replay_one_graph` 没有保存中间解码得到的最大值，**刻意保持原 Baseline 的“按轨迹能量最优档案→A.3”规则**。后续要评估 PAVE 时，再让同一个冻结 PAVE 接到 replay 返回的真实节点集合上；不能悄悄给其中一组增加全历史解码。

### 5.5 四个干预组及必须包含的额外对照

| 干预组 | 现有 F4 参数 | 修改什么 | 检验哪条假设 |
|---|---|---|---|
| I0 Full | 不改 | 原 low＋high＋两门控 | 参照 |
| I1 No-Low | 同 I0 | `low_override=False` | low 分支是否有负作用 |
| I2 No-High | 同 I0 | `high_override=False` | high 分支是否有负作用 |
| I3 Gate=0.5 | 同 I0 | `gate_override=0.5` | 状态门控与简单固定融合差异 |
| I4 No-Adapter | 同 I0 | `V4FrequencyGCN(...,disable_adapter=True)` | 同一个骨干训练权重下的纯适配器贡献 |
| I5 αL/αH 扫描 | 同 I0 | 不改权重，仅评估 α 的固定比例 | 残差强度是否过强／过弱 |
| C1 Original RLNN | v2 `none` | 独立原权重 | 真正算法起点 |
| C2 Message | v4 `v4a_message` | 同信息普通消息适配器 | 频域参数化独立贡献 |
| C3 State MLP | v4 `v4a_mlp` | 同容量统计适配器 | 是否只需要一般非线性容量 |

**残差扫描建议**：沿已训练 `α_low`、`α_high` 同比例乘 `s∈{0,0.25,0.5,1,1.5,2}`；这是固定权重敏感性诊断，不能在验证集上选最佳 `s` 后直接当作无调参的新方法。若研究两路交互，可另设 `sL,sH ∈ {0,0.5,1}` 的 3×3 小网格，先在探索子集测完再复核。

**禁止误读**：现有 `low_override=False` 或 `high_override=False` 虽让输出置零，但仍经过相应 MLP 计算。**其推理时间不能用于估计真的删掉该分支后会加速多少**；性能干预与速度优化必须分开。

### 5.6 在“同状态”里应记录什么？

对每个 `graph_id, t, particle, model, intervention`：

- **翻转概率变化**：`mean |p_I−p_full|`、p90、p99、节点 top-5 翻转概率集合 Jaccard；
- **方向性**：哪些节点类别概率升/降（见第 6.2 节的分组定义）；
- **距离正则**：`∑_i p_i` 距目标 d=5 的误差，避免模块只是让翻转更少而“看起来稳定”；
- **即时能量**：在同状态下计算每个单节点翻转的精确 `ΔE`，检查概率是否倾向于能量有利的动作（这是局部诊断，不等于未来收益）；
- **冲突**：当前 selected-selected 边数、可直接加入节点数、唯一阻挡节点数；
- **差异集中程度**：`Δlogit` 是否只集中在一小批节点还是整个图一致平移。

**同状态下概率变化本身不算收益**。它必须与 P3 的真实成对回报对照。

### 5.7 统计口径和最小实验规模

**试运行**：10 张训练图 × seed0 × I0–I5 × t=25/50/100，确保样本、配对状态和保存完整。

**复核**：100 张验证图 × 3 seeds × I0–I4；同状态诊断先用 4 个粒子，完整 replay 保留 K=20/T=100。完整 replay 的 1000 图正式结果本轮不需要。

输出要分成两张表：

**表 A：固定相同状态的改变**（`Δp`、expected flips、单步有利动作相关性）。

**表 B：固定 `x0,U` 的完整求解结果**（Mean Size、逐图 Δ、胜平负、曲线、时间只记录诊断运行时间）。

按 `graph_id×train_seed` 配对计算差值，不能把 20 个粒子当成 20 个独立图；3 个种子的均值±样本标准差单独报告。可以在 100 张图单位上做配对 bootstrap 置信区间，但不要把它描述为正式独立测试显著性。

---

## 6. P3：真实动作收益诊断，怎样修改 `probe_v4_utility.py`？

### 6.1 为什么现有排序正确率 48.84% 不能直接判死刑

仓库现有探针固定老师、每个节点动作重复 **R=2 次**，20 张验证图、三个搜索时刻、每个状态随机挑 16 个节点。现有记录指出，两次都非平局的动作比较中，方向一致性约为 **52.64%**，标签本身明显不稳定。

此外，**RLNN 每步可以同时翻转多个节点**。单节点“翻或不翻”的 H=20 未来收益排序，是一种有价值的受控分析，**不等于**原模型必须把所有 logits 按该排序排列。因此 48%–49% 的排序正确率并不直接说明训练实现错误，只说明当前没有观察到独立的收益排序优势。

这一次的目标是让标签更稳定、样本更有代表性，并区分：

- 各模型对**同一候选动作集合**的分数排名；
- 多次续跑平均的**实际解码收益**；
- 由随机性产生的收益标签不确定度；
- 频率方法是否比普通消息和原始 RLNN 在**特定节点类别**更有效。

### 6.2 将节点随机抽样改为“保留自然分布＋分层扩充”

当前：

```python
candidates = torch.randperm(graph.num_nodes, generator=candidate_rng)[:args.nodes]
```

建议保留一组原始 **Uniform16** 作为与历史结果可比的对照，再新增 **Stratified16/24**：

| 候选类别 | 规则 | 为什么重要 |
|---|---|---|
| A. 已选且冲突 | `x_i=1, c_i>0` | 约束修复，移除单节点可能消除冲突 |
| B. 已选且不冲突 | `x_i=1, c_i=0` | 单步删除会降低基数，可能为多步交换让路 |
| C. 未选、可直接加入 | `x_i=0, c_i=0` | 典型的一步增益 |
| D. 未选、唯一阻挡 | `x_i=0, c_i=1` | 与 v3 平台搜索的 1→1/后续修复相关 |
| E. 未选、多重阻挡 | `x_i=0, c_i≥2` | 可能需要组合动作，不应与 D 混同 |
| F. 释放机会偏高 | 根据 `r_i`、唯一阻挡释放计数选择 | 检验分频是否对交换机会敏感 |

`c` 使用 `topology.binary @ x`（单位边权），`r` 使用当前 `state_bands()`，不要错用带 0.5 权重的能量邻接。

每组优先抽若干节点，不足则从其他组补齐；全部按预先固定的 `candidate_seed` 去重并排序。保存每组**可选总数、最终选中数、抽样比例**，禁止因为分层过采样而把“有用动作占比”误报为自然节点总体比例。

建议分别输出：Uniform16 的自然分布指标、Stratified24 的困难动作指标。只有样本设计完全相同的方法可以放入同一张排名表。

### 6.3 让续跑重复次数 R 可配置，不能只用 2

在 `models/probe_v4_utility.py::main()` 增加：

```python
parser.add_argument("--replicates", type=int, default=8)
parser.add_argument("--sampling", choices=("uniform", "stratified"),
                    default="uniform")
parser.add_argument("--phases", nargs="+", type=int, default=[25, 50, 100])
```

并在 `probe()` 调用：

```python
sizes, gains = paired_utilities(
    teacher, x[:, 0], candidates, graph, adj, normalized, problem,
    args.horizon, seed=node_seed + 1000000,
    replicates=args.replicates,
)
```

**必须同步重构输出格式**：现有 `actions.csv` 仅硬编码 `gain_r0/gain_r1` 和 `flip_size_r0/r1`，不能只改参数就把 R=8 写入原 CSV。建议将结果拆成：

```text
action_replicate.csv:
  graph_id,state_id,node,category,replicate,
  noflip_size,flip_size,gain,continuation_seed

action_aggregate.csv:
  graph_id,state_id,node,category,R,
  gain_mean,gain_std,gain_sem,positive_fraction,
  negative_fraction,zero_fraction,logit_B,logit_MSG,logit_F4

ranking.csv:
  graph_id,state_id,model,R,candidate_sampling,
  informative,non_tied_pairs,pair_accuracy,top1_gain,
  uniform_gain,oracle_sample_gain,top1_regret
```

`gains` 形状保持 `[R,num_candidates]`；`mean_gain=gains.float().mean(0)`，样本标准差 `gains.float().std(0, unbiased=(R>1))`，标准误 `std/sqrt(R)`。对完全相同的状态和候选集合，所有模型共享这套收益标签，只改变评分分数。

### 6.4 成对随机数怎么生成才公平？

当前 `paired_utilities()` 已经让同一个 replicate 中的“翻转”和“不翻转”共享随机续跑序列，应该保留。建议只把随机种子生成方式明确成：

\[
U_{g,s,t,i,r,h,v}\quad (r=1,\ldots,R),
\]

其中 `g` 图、`s` 状态、`t` 观察时刻、`i` 候选动作、`r` replicate、`h` 续跑步、`v` 节点。**同一 replicate 内，所有候选动作与 no-flip 分支共用相同的 h×v 噪声；不同 r 使用相互独立的确定性种子**。

可以使用固定映射，例如 `base_seed=10_000_003*graph_id + 1009*t + 37*state_id` 再派生各 replicate；具体整数如何编码不重要，必须稳定、不碰撞且全部写入 manifest。不同模型不能因为自身 logits 的计算次数而多消耗随机数。

这是**共同随机数（CRN）配对**，用于降低动作比较方差；并不消除有限 rollout 的随机误差，也不是识别现实世界的因果后门。

### 6.5 先确认收益标签可靠，再计算模型优劣

定义第 r 次重复中动作 i 的合法解收益：

\[
Y_{s,i,r}=\left|\mathrm{Decode}\bigl(\mathrm{Archive}(X^{\mathrm{flip}(i)}_{t:t+H,r})\bigr)\right|
-\left|\mathrm{Decode}\bigl(\mathrm{Archive}(X^{\mathrm{keep}(i)}_{t:t+H,r})\bigr)\right|.
\]

\[
\widehat{U}_{s,i}=\frac1R\sum_{r=1}^{R}Y_{s,i,r}.
\]

这里的 archive 是**原 RLNN 每粒子历史最低能量状态记录**，并非已经放弃的创新点一。最外层还可以报不经过 PAVE 的 A.3 解码收益；**不能把 PAVE 训练成对照的监督标签**。

先报告以下**标签可靠性**：

- 每状态候选中 `Y=0` 的比例、`\widehat U` 的标准误、正收益候选占比；
- R=8 的前 4 次、后 4 次各自平均收益排名的 Spearman/Kendall 或 top1 一致率；
- 两组独立重复中收益方向一致率、非平局候选对数量；
- 状态是否 informative（至少有一对候选的真实平均收益不同），比例是多少；
- R 从 2→4→8→16 时，排名指标是否稳定。**先在小子集做 R 收敛，不用所有图都做 R=16。**

若 R=8 仍大量不稳定，不应盲目增大神经网络；先检查是否是 H=20 续跑噪声过大，尝试 H=10/20/40 的**机制诊断对照**（其中 H=40 需要给 CLI 扩充选择并记录计算成本）。不同 H 是不同目标，不放进同一张没有区分的主表。

### 6.6 再评价“模型是否利用了这些动作信息”

统一的分数是各模型同状态下的**flip logits**。比较 B、B+FT、MLP、MSG、F4，并加入：

- **即时能量改变量** `−ΔE_flip` 的解析排名对照；
- **随机排名**（同一候选集固定种子）；
- **仅度数／选中邻居数／释放数** 的简单规则基准；
- **F4 关闭 high、关闭 low** 后在同一个 `x` 上的排名。

逐状态的指标：

\[
\mathrm{Top1Gain}=\widehat U_{s,\arg\max_i\mathrm{logit}_i},\qquad
\mathrm{Regret}=\max_i\widehat U_{s,i}-\mathrm{Top1Gain}.
\]

另测候选对排序正确率（只统计非平局对）、按状态等权的排序正确率、按候选对数量加权的排序正确率、top1 选中正收益动作的比例。**状态间候选对数量不均，必须同时报告等权和加权口径**。

真正检验“频域独立价值”的主对照是：

\[
\Delta\mathrm{Regret}=\mathrm{Regret}_{MSG}-\mathrm{Regret}_{F4},
\]

在同一图、同一状态、同一候选动作、同一收益标签的条件下，`ΔRegret >0` 才说明 F4 选得更接近标签中的最好动作；还必须看各模型排序差异是统计稳定还是偶然。比较**绝对 logits 数值**没有意义，参数化不同可有任意单调变换；排序与实际收益才更有价值。

**关键限制**：单节点强制翻转的 H=20 续跑，与 RLNN 一步可同时翻转多个节点的联合 Bernoulli 决策并不相同；它只能作为诊断相关性，不能直接成为“主模型应当按 Top1Gain 优化”的证明。

### 6.7 建议具体执行批次

| 批次 | 图数／种子 | t | 节点抽样 | R | 目的 |
|---|---|---|---|---:|---|
| P3-0 冒烟 | 3 图、seed0 | 25 | Uniform8 | 2 | 检查输出、有限性、随机复现 |
| P3-1 标签可靠性 | 10 图×3种子 | 25/75 | Uniform16 | 2、4、8 | 检查收益稳定所需重复次数 |
| P3-2 节点类型 | 20 图×3种子 | 25/50/100 | Uniform16 与 Stratified24 | 8 | 研究高频是否在关键类别上有优势 |
| P3-3 复核 | 最多100图×3种子 | 25/50/100 | 仅保留已发现有效的抽样定义 | 8 | 复核趋势，不用于开发测试集调参 |

样本规模很大时，每组先记录 CPU/GPU 实际开销，再决定是否扩图数。**没有理由在前 10 图证据明显为零时，立即计算 100 图×R=16 的所有节点。**

### 6.8 对 `probe_v4_utility.py` 的额外安全检查

当前工具里 `teacher` 默认原 RLNN，且继续策略冻结。如果模型 logits 是同一状态上不同模型给出的分数，那么标签仍是**原教师下**的收益，不能解释成“F4 自身策略未来能获得的收益”。建议分别明确两种任务：

- `teacher_continuation`（第一阶段主诊断）：所有模型共享标签，严格隔离表示/评分差异；
- `model_own_continuation`（仅后续解释）：分别以 B、MSG、F4 的策略续跑，比较算法整体表现，标签已不同，不能将回报差只归因于同一动作评分。

把 `continuation_policy` 写入 run.json，禁止混合汇总。PAVE 一律不进入这两种动作收益标签计算。

---
## 7. P4：计算成本到底耗在哪里？如何独立剖析？

### 7.1 当前已经知道什么

v4 记录显示，在 FP16 固定 K20/T100 的验证协议下，原骨干无 PAVE 约 **0.31150 s/图**，F4 无 PAVE 约 **0.52346 s/图**；同 0.5 秒并接冻结 PAVE，原骨干约搜索 **98.12 步**，F4 约 **59.09 步**。这说明每步成本是实际瓶颈，且超出的耗时足以显著改变时间受限时的搜索预算。

**已知拓扑并不是每步重新创建**：`models/eval_v2.py::search()` 每图构造一次 `solver.policy_topology(graph,adj)` 并传递给 `sample_states`；故不能未经测量就把全部耗时归咎于邻接矩阵反复建立。真正应测的是**每步额外状态依赖稀疏传播、MLP/LayerNorm/gate 和 Python 调度**。

### 7.2 计算图拆解

当前 F4 每一次翻转概率计算，大致包含：

```text
当前状态 x [N,K]
   │
   ├─二值邻接 A @ x                         → c  (1 次邻域聚合)
   ├─A @ release                           → r  (1 次)
   ├─状态信号 Z=[x,log1p(c)]
   ├─P @ Z                                 → PZ (1 次邻域聚合)
   ├─P @ PZ                                → P²Z (1 次)
   ├─low MLP + LayerNorm                   → u_low
   ├─high MLP + LayerNorm                  → u_high
   ├─6维门控 + 残差系数                    → delta
   └─原 5 层 GCN                           → 5 次 GCN 稀疏邻域传播
                                                ↓
                                       翻转 logits
                                                ↓
                                      Bernoulli更新
```

上面列的是**逻辑操作数**，不等于已经实测到 9 次完全同成本的 GPU kernel；稀疏算子可能拆分为多个 kernel，形状差异也会影响成本。

### 7.3 新增 `models/profile_v4_frequency.py`

不要在 `diagnose_v4_frequency.py` 的 observer 内插入阶段耗时后直接与原始 Time 对比：诊断里额外有特征分解、全图特征值求解和多阶段解码，会严重改变耗时。应该独立构建 profiler：

1. 每次只启动一个评测进程，固定一张 RTX 5090；另一张卡若有其他训练，不用于横向时间结论。
2. 先完成 20–50 次预热，避免首图 CUDA 初始化和 JIT/内存分配误差。
3. 选 20 张固定图，覆盖当前 RB200–300 节点/边数的不同区间；对每图固定 K=20、T=100。
4. 先测真实端到端单图时间，再测组件微基准，二者分开保存。
5. 每组重复至少 5 次，记录中位数、p90/p95、MAD/标准差；实验次序在方法间交替，避免持续显卡温度／时钟漂移造成偏差。

建议明确计时边界：

```text
P4-Total: 服务器内数据已上 GPU后的同步点
   ├─图拓扑构造（每图一次）
   ├─100次每步采样与状态更新
   │   ├─state_bands:  c/release/r
   │   ├─lazy_walk:   PZ/P²Z
   │   ├─encoder_low
   │   ├─encoder_high
   │   ├─gates/alphas/fusion
   │   ├─original_5layer_GCN
   │   └─Bernoulli sample + energy update
   ├─best-energy archive + A.3 decode
   └─冻结 PAVE（若启用）
P4-Total 不计数据读取、初始 H2D、训练和权重载入
```

注意原 `v4a_mlp` 和 `v4a_message` 也会经过 `state_bands` 构造共享门控特征；公平 profiling 时需测它们**实际代码路径**，不要误以为 MLP 对照完全没有图传播。

### 7.4 建议的 GPU 时间工具

方法一：**端到端计时**使用原仓库 `perf_counter`、CUDA 同步和 `eval_v4_frequency.py`。这个口径是论文主表依据。

方法二：**单组件计时**可以使用带同步的函数包裹器（用于粗剖析），或 CUDA Event 对 GPU 部分计时：

```python
from time import perf_counter
import torch

def synchronized_wall(fn, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, perf_counter() - start


def cuda_event_ms(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    end.synchronize()
    return result, start.elapsed_time(end)
```

**不要将每个组件单独同步后的耗时直接相加，当作真实一次完整推理的时间**：同步本身会改变执行调度，也可能掩盖异步重叠。应用 profiler（`torch.profiler`，只在少量图上启用）定位 kernel、CPU 发射、Sparse SpMM 和 Python 开销，性能主表仍以无 profiler 的真实 E2E 为准。

### 7.5 优化建议按“是否改变结果”分类

| 优化 | 是否影响数学输出 | 验证要求 |
|---|---|---|
| 复用每图拓扑 CSR、度数、GCN 归一化 | 理论不变；当前已有缓存 | 检查没有每步重复创建拓扑，不能重复计功 |
| 将相同的 `A@x` 在 state 与 gate 中复用 | 理论不变 | FP32 差异阈值内结果相同；FP16 单独容差检验 |
| 合并可兼容的稀疏传播或重排维度 | 理论不变，浮点归约顺序可能变 | 测 logits、轨迹和最终 Size，不要求浮点位级相同 |
| 消除重复 `.flatten/.reshape` 与临时张量 | 理论不变 | 测真实设备/带宽改善 |
| 精简 high MLP、删除频带或压低阶数 | **改变模型** | 必须重训练同参数／同预算对照，不能只报速度 |
| 每隔 k 步重算高低频 | **改变状态响应** | 必须明确为近似策略，测滞后误差与搜索退化 |
| 仅在困难节点计算频率分支 | **改变模型/采样策略** | 必须明确困难节点筛选成本，检查是否漏掉有利节点 |
| `torch.compile`、CUDA Graph | 可能改变算子/图行为 | 先验证稀疏与动态形状兼容，再检查实际开销与数值 |

**先做不改变数学输出的优化；只在 P3 证明任务收益时再考虑改变结构节省时间。**

### 7.6 应该出具的效率表

| 模型 | Topology ms/图 | 状态统计+滤波 ms/100步 | 两路编码+gate ms/100步 | 原 GCN ms/100步 | A.3 ms/图 | PAVE ms/图 | E2E ms/图 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B | 待测 | — | — | 待测 | 待测 | 待测 | 待测 |
| MLP | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| MSG | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| F4 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

再绘制 **Size–总时间 Pareto 图**，并标注每种方法在 0.5 秒内真实完成的搜索步数。只有同预算图才能说明效率优劣。

---

## 8. 具体怎么运行？现有命令与需要新写的命令分开

### 8.1 运行前设置

在服务器项目根目录：

```bash
cd /home/hzeng/project/ZH/RLNN_lab
export PYTHON=/home/hzeng/envs/rlnn/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WANDB_MODE=disabled
mkdir -p outputs/v4_frequency/followup/{p0_checks,p1_spectrum,p2_branch,p3_utility,p4_profile,analysis}

git rev-parse HEAD
$PYTHON -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))'
```

**先核对这份文档的 SHA**。若服务器后来又提交了改动，先记录新的 SHA 和差异，再执行，不要把不同版本混入同一次复现实验。

检查权重存在和冻结 PAVE 的 SHA256（运行服务器本地命令，不上传权重）：

```bash
for seed in 0 1 2; do
  test -f "outputs/v2/none_seed${seed}/checkpoints/best.pt" || exit 1
  test -f "outputs/v4_frequency/v4a_seed${seed}/checkpoints/best.pt" || exit 1
  test -f "outputs/v4_frequency/v4a_message_seed${seed}/checkpoints/best.pt" || exit 1
  test -f "outputs/v4_frequency/v4a_mlp_seed${seed}/checkpoints/best.pt" || exit 1
  test -f "outputs/v3/stats_h1_seed${seed}/checkpoints/best.pt" || exit 1
  sha256sum "outputs/v3/stats_h1_seed${seed}/checkpoints/best.pt"
done
```

每次独立运行的 metadata 写明：`git_sha, torch, cuda, gpu, precision, graph_ids, model_path, model_sha256, pave_sha256, seed, actual_K, actual_T, num_d, beta, postprocess, time_budget`。

### 8.2 现有命令：先重画原始频率诊断（不用改代码）

以下命令是仓库已经支持的。特别注意 `eval.sh` 默认 `SPLIT=test`，所以**必须显式设置 `SPLIT=validation`**，避免误用官方测试集选择新方案。

```bash
# 先用 seed0 20张验证图复查 F4 的原始频谱
CUDA_VISIBLE_DEVICES=0 SEED=0 SPLIT=validation SAMPLES=20 \
  SPECTRAL_GRAPHS=20 PRECISION=no \
  RUN_NAME=followup_original_diag_f4_seed0 \
  bash train_scripts/v4_frequency/diagnose.sh v4a

# 同样条件跑普通消息，注意使用匹配的seed0权重
CUDA_VISIBLE_DEVICES=0 SEED=0 SPLIT=validation SAMPLES=20 \
  SPECTRAL_GRAPHS=20 PRECISION=no \
  RUN_NAME=followup_original_diag_message_seed0 \
  bash train_scripts/v4_frequency/diagnose.sh v4a_message
```

**已有文件只需直接分析时，甚至不用重跑**：

```text
outputs/v4_frequency/diagnostic_v4a_seed{0,1,2}/
  bands.csv, spectra.csv, conditional_distributions.csv,
  energy_conflict_curves.csv, gate_histograms.json, run.json

outputs/v4_frequency/utility_seed{0,1,2}/
  actions.csv, ranking.csv, states.csv, run.json
```

优先读取这些已有记录，避免重复出大量完全相同的数据。

### 8.3 新增 P1 命令（**新增脚本实现后才可运行**）

在 `models/diagnose_v4_frequency.py` 加 `--trace-encoder`、`--spectral-center` 标志，并将 `--result_dir` 写入 followup 路径。修改已有 `train_scripts/v4_frequency/diagnose.sh` 末尾传参机制已经支持 `"$@"`，因此可直接：

```bash
CUDA_VISIBLE_DEVICES=0 SEED=0 SPLIT=validation SAMPLES=20 \
  SPECTRAL_GRAPHS=20 PRECISION=no RUN_NAME=followup_p1_f4_seed0 \
  bash train_scripts/v4_frequency/diagnose.sh v4a \
    --trace-encoder --spectral-center
```

**注意**：上述两个新 CLI 选项当前不存在；必须先在新代码里 `parser.add_argument(...)`，然后让 `diagnose()` 根据开关保存新 CSV。`RUN_NAME` 默认把输出放在 `outputs/v4_frequency/` 根目录，不会自动进入 `followup/p1_spectrum/`；可在新增脚本中将 `--result_dir` 改为绝对目录，或在结果发布时按 manifest 移动，不要与历史实验同名覆盖。

### 8.4 新增 P2 命令（**新增 `diagnose_v4_branch_replay.py` 后才可运行**）

新 CLI 建议支持：

```text
--split validation
--test-path models/data/MIS_tensors/RB_small_val
--seed 0|1|2
--graphs 100
--num-k 20
--num-t 100
--precision fp32|fp16
--checkpoint-base ...
--checkpoint-f4 ...
--checkpoint-message ...
--checkpoint-mlp ...
--interventions full no_low no_high gate_half no_adapter
--mode same_state|replay|both
--result-dir outputs/v4_frequency/followup/p2_branch/seed0
```

调用模板：

```bash
CUDA_VISIBLE_DEVICES=0 $PYTHON -u models/diagnose_v4_branch_replay.py \
  --split validation --graphs 100 --seed 0 \
  --num-k 20 --num-t 100 --precision fp32 \
  --checkpoint-base outputs/v2/none_seed0/checkpoints/best.pt \
  --checkpoint-f4 outputs/v4_frequency/v4a_seed0/checkpoints/best.pt \
  --checkpoint-message outputs/v4_frequency/v4a_message_seed0/checkpoints/best.pt \
  --checkpoint-mlp outputs/v4_frequency/v4a_mlp_seed0/checkpoints/best.pt \
  --interventions full no_low no_high gate_half no_adapter \
  --mode both --result-dir outputs/v4_frequency/followup/p2_branch/seed0
```

这里是**应实现的接口契约**，不是仓库当前已有的可直接运行命令。新增脚本必须先通过 P0 单元测试和 3 图冒烟。

### 8.5 新增 P3 命令（需要先完成 R 与 CSV 格式改造）

历史可运行的原命令：

```bash
CUDA_VISIBLE_DEVICES=0 SEED=0 MODEL_SEED=0 GRAPHS=20 NODES=16 \
  RUN_NAME=original_probe_seed0 \
  bash train_scripts/v4_frequency/probe_utility.sh
```

新设计命令：

```bash
CUDA_VISIBLE_DEVICES=0 SEED=0 MODEL_SEED=0 GRAPHS=20 NODES=24 \
  RUN_NAME=followup_utility_seed0_stratified_r8 \
  bash train_scripts/v4_frequency/probe_utility.sh \
    --replicates 8 --sampling stratified --phases 25 50 100
```

新参数和分层抽样当前都没有实现，须先修改 `models/probe_v4_utility.py`；原 shell 本来有 `"$@"`，因此不需要改原脚本传参逻辑。输出必须拆分逐 replicate 与统计聚合，否则旧字段 `r0/r1` 会丢数据。

### 8.6 P4 命令（**新增 profiler 后才可运行**）

```bash
CUDA_VISIBLE_DEVICES=0 $PYTHON -u models/profile_v4_frequency.py \
  --seed 0 --graph-count 20 --warmup 30 --repeats 5 \
  --models baseline mlp message f4 \
  --precision fp16 --num-k 20 --num-t 100 \
  --result-dir outputs/v4_frequency/followup/p4_profile/seed0
```

建议每个 profiling 阶段独占一张 GPU，避免上一轮 v4 训练时“同卡并行 6 进程”的负载污染速度结论。`--warmup` 和 `--repeats` 是新实现选项，不是现有脚本的现成参数。

### 8.7 做完整 E2E 实验时，复用原评估而不是自己造新计时器

诊断完成后，如果需要再次查看完整算法结果，运行**已存在**的接口：

```bash
# F4，100张验证图、FP16、K20/T100，同一次运行保存none和冻结PAVE
CUDA_VISIBLE_DEVICES=0 SEED=0 SPLIT=validation SAMPLES=100 PRECISION=fp16 \
  RUN_NAME=followup_e2e_v4a_seed0 \
  bash train_scripts/v4_frequency/eval.sh v4a
```

这条命令沿用当前 `eval_v4_frequency` 的 PAVE 冻结约束。如果另要 0.5 秒预算，使用现有 `eval_budget.sh` 并显式设定 `SPLIT=validation SAMPLES=100`，一次只能有一个 `--post` 策略。

**禁止**用 `diagnose.sh` 测出来的完整墙钟时间替代 E2E Time，诊断观察器增加了大量额外工作。

---

## 9. 结果汇总：论文式主表、诊断表和图应该怎么做

### 9.1 评价分母不要搞错

- **Mean Size**：每张图在 K 条轨迹中选最好合法结果，然后对图平均。
- **训练种子 Std**：每个 seed 在固定图集上的 Mean Size 是一个数，三种子再算样本标准差。
- **逐图配对**：每图每种子计算 `F4−Message`、`F4−MLP`、`F4−B`，再报平均 ΔSize、胜／平／负；跨图差应图级聚合。
- **分支干预**：相同权重、相同 `x0,U`；统计的是干预算法而不是重训练模型。
- **候选动作收益**：每个状态的候选动作不独立；R 次重复是 Monte Carlo 估计，不等于 R 个独立图。排序指标先按状态计算，再按图、再按 seed 汇总。
- **时间**：分开“模型本身求解时间”和“诊断额外开销”，只把前者写进论文主质量-时间表。

### 9.2 推荐的四张定量表

**表 1：频域层级机制（P1）**

| 模型 | 阶段 | 原始高频能量% | 去驻留后高频能量% | 驻留占比% | 绝对能量 | Δlogit RMS |
|---|---|---:|---:|---:|---:|---:|
| F4 | `high_input` | 待测 | 待测 | 待测 | 待测 | — |
| F4 | `high_Linear1` | 待测 | 待测 | 待测 | 待测 | — |
| F4 | `high_SiLU` | 待测 | 待测 | 待测 | 待测 | — |
| F4 | `high_Linear2` | 待测 | 待测 | 待测 | 待测 | — |
| F4 | `high_LayerNorm` | 待测 | 待测 | 待测 | 待测 | 待测 |
| MSG | 对应第二分支输出 | 待测 | 待测 | 待测 | 待测 | 待测 |

必须说明“高频能量”的阈值及图谱支持。不同阶段通道数不同，不可仅凭总绝对能量大小说好坏。

**表 2：同状态与完整 replay 干预（P2）**

| F4 干预 | 同状态平均Δp | 同状态期望 flips | Replay Mean Size | ΔFull | 胜/平/负 |
|---|---:|---:|---:|---:|---|
| Full | — | 待测 | 待测 | — | — |
| No-Low | 待测 | 待测 | 待测 | 待测 | 待测 |
| No-High | 待测 | 待测 | 待测 | 待测 | 待测 |
| Gate=0.5 | 待测 | 待测 | 待测 | 待测 | 待测 |
| No-Adapter（同权重） | 待测 | 待测 | 待测 | 待测 | 待测 |

**表 3：动作收益排序（P3）**

| 模型 | informative 状态数 | 等权 pair accuracy ↑ | top1 增益 ↑ | regret ↓ | 按关键节点组的优势 |
|---|---:|---:|---:|---:|---|
| B | 待测 | 待测 | 待测 | 待测 | — |
| 即时能量规则 | 待测 | 待测 | 待测 | 待测 | 待测 |
| MSG | 待测 | 待测 | 待测 | 待测 | 待测 |
| MLP | 待测 | 待测 | 待测 | 待测 | 待测 |
| F4 | 待测 | 待测 | 待测 | 待测 | 待测 |

R=2、4、8 的标签稳定性另做小表；若排序标签不可靠，应把“无法判断”写成正式诊断结果，不能强行解释 F4 的 1–2 个百分点差异。

**表 4：完整求解质量／时间（P4）**

| 方法 | 固定 K20/T100 Size↑ | Time↓ | 0.5s Size↑ | 0.5s 实际步数 | Params | 可行率 |
|---|---:|---:|---:|---:|---:|---:|
| 原骨干 | 已有 | 已有 | 已有 | 已有 | 165,505 | 100% |
| MLP | 已有 | 已有 | 已有 | 已有 | 170,513 | 100% |
| MSG | 已有 | 已有 | 已有 | 已有 | 170,513 | 100% |
| F4 | 已有 | 已有 | 已有 | 已有 | 170,513 | 100% |
| 原骨干＋冻结PAVE | 已有 | 已有 | 已有 | 已有 | 169,218 | 100% |
| F4＋冻结PAVE | 已有 | 已有 | 已有 | 已有 | 174,226 | 100% |

这里“已有”应引用当前 v4 验证结果，不要用后续测试模型覆盖，诊断的时间不填进表 4。

### 9.3 推荐的五张图

1. **频率逐层 Sankey/堆叠图**：`high_input→Linear1→SiLU→Linear2→LayerNorm→delta_high→logits`，展示原始及去驻留后频谱变化。
2. **节点状态条件图**：`c=0/1/≥2 × selected/unselected`，门控 `gL/gH` 与 `Δp` 分布。
3. **动作收益可靠性图**：R=2/4/8 时状态内 top1 收益估计的分布、收益排序稳定性。
4. **逐状态 paired scatter**：横轴 MSG regret，纵轴 F4 regret，点按节点类别和 t 上色；对角线下方意味着 F4 更好。
5. **Size–时间 Pareto 图**：固定工作量和0.5秒两组分开，标注真实搜索步数及后处理耗时。

图中应以图为聚合基本单位；把每个节点直接画成独立统计样本，只可用于探索可视化，不可据此计算图级可信区间。

---

## 10. 结果怎么解释？预先确定的判断规则

### 10.1 把证据分成四条线

| 证据线 | 支持继续频域研究的表现 | 应当停止／重设假设的表现 |
|---|---|---|
| 频率机制 | 找到具体的有信息阶段，且该阶段对 logits/关键节点有可靠影响 | 只发现比例变化，却不关联任何有利动作或最终 Size |
| 同状态干预 | 高/低频分支或其交互在多个 seed、多个图上有一致的有益行为 | 关闭支路或换成普通消息没有损失，甚至更好 |
| 成对收益 | F4 在**同状态、同候选、同回报标签**下比 MSG 有稳定更低 regret | F4 排序无优势，或收益标签随机性过大无法识别 |
| 总时间 | 在同时间预算下收益成立，或已定位可消除的开销 | 等步数已更差、时间又明显更慢，且无可消除的任务损失 |

### 10.2 推荐采用的阶段门槛（研究筛选标准，非统计定理）

**门槛 D1——可靠信号**：R=8 的标签在主要动作类别上具备可重复的方向或排序信息；若重复间一致性近似随机、非平局覆盖过低，记为 `inconclusive`，先改测量，不能宣布有效或无效。

**门槛 D2——频域独立优势**：在图级配对下，F4 相比 MSG 的 regret/排序指标方向为正，至少 2/3 seeds 一致；效应量不只是极少数状态造成。建议报告以图为 bootstrap 单位的区间；本轮验证图已用于结构选择，区间只是诊断不代表独立确认。

**门槛 D3——完整求解优势**：现有 100 图 FP32/FP16 固定工作量比较至少与 MSG 持平，新的候选机制改动才可能进入短训；若同样搜索步数已更差，单纯减少时间不能解释全部失败。

**门槛 D4——部署预算优势**：若改了结构并训练后，必须在 0.5 秒/图下至少不弱于**原骨干＋同一冻结 PAVE**，并与**原骨干持续搜索＋PAVE**这一强对照报告差异。不能仅与没有后处理的原 RLNN 比较。

这些是**预先承诺的工程筛选门槛**。不要为了让某个模块过关，实验结束再改阈值、抽样规则、观测时刻或比较对象。

### 10.3 四种可能结论及下一步动作

**情形 A：频域信号有用，但 high 编码后丢失有利信息。**

只有当 P1 观察到明确的信息丢失，且 P2/P3 证明这与有利动作决策有关时，才研究**频率保真残差路径**：例如 high 分支采用跳过非线性的大通路、将高频线性项直达翻转 logit、或只对高频内容做轻门控。必须与同参数消息对照一致，并用单分支和双分支重训练验证。**不能仅为了让谱能量看起来更高而加频域正则。**

**情形 B：高频收益只集中在少数节点和时刻。**

研究**事件触发的稀疏频域路径**：只有当冲突、唯一阻挡或交换机会满足条件时计算额外分支，并量化筛选成本。不能只在有效候选集上报告收益，而忽略识别候选本身的计算代价。

**情形 C：同状态有利但完整时间预算失败。**

优先优化当前实现中的重复稀疏邻域聚合、跨步共享、张量物化与 CPU/GPU 同步，而不是新增更多滤波尺度。比较“同数学模型、只改变实现”的前后结果和 Time；若优化不足，再讨论结构剪枝。

**情形 D：频域对动作收益、同工作量结果和强时间基线均无独立优势。**

**如实终止这一代频率模块。** 此时第二个创新点需要重定义问题，而不应继续叠加注意力、多尺度波段、复杂损失，去追求随机性的几百分之一 Size 改善。PAVE 保持独立可用。

---

## 11. 实际开发任务拆分：一条条做，不返工

### 任务 1｜冻结环境并记录 manifest

**修改文件**：无需改模型；写 `followup/manifest.json`。  
**完成标准**：SHA、设备、权重 hash、图 ID 与种子齐全，PAVE hash 全程一致。

### 任务 2｜扩展谱诊断纯函数

**新增**：`models/frequency_diag_utils.py`。  
**修改**：`models/diagnose_v4_frequency.py`，增加 CLI、逐层跟踪 CSV、去驻留统计。  
**完成标准**：逐层跟踪值与原 `adapter.high/low` 输出一致，频带能量总和在数值误差范围内，图重编号后谱统计不变。

### 任务 3｜做 P1 三图冒烟，再做20图探索和三种子复核

**修改文件**：只需 run 脚本、汇总脚本。  
**完成标准**：明确哪个阶段造成 raw/centered 频率占比变化；能区分纯驻留影响与非驻留影响；不把相关性叫原因。

### 任务 4｜实现 P2 同状态／共享噪声 replay

**新增**：`models/diagnose_v4_branch_replay.py`。  
**修改**：`models/mis_v4_frequency.py` 可选增加无参数 override context manager；或者把它放在诊断脚本中。  
**完成标准**：同模型同初始状态重复 replay 完全复现；模型 I0 与直接正常计算在固定状态下等价；干预结果既有 same-state 又有 final Size。

### 任务 5｜重构收益探针的数据格式与采样

**修改**：`models/probe_v4_utility.py`。  
**新增**：逐 replicate 输出、分层采样、R CLI、收益标签稳定性统计。  
**完成标准**：R=2 使用 uniform 时保持原有数值口径；R=4/8 数据结构正确，`gain_mean` 与逐重复均值完全一致。

### 任务 6｜P3 标签可靠性先行，再研究动作排序

**训练**：不训练。  
**完成标准**：给出收益标签稳定性、类别分层指标、F4 相比 MSG/MLP/B 的 regret 与逐图差，能判断排序变化是否超过噪声。

### 任务 7｜P4 独占 GPU 耗时剖析

**新增**：`models/profile_v4_frequency.py`。  
**完成标准**：分清 topology／state bands／encoder／gate／GCN／A.3／PAVE 的成本；解释 F4 同预算步数减少；不能以被 observer 污染的计时写结论。

### 任务 8｜汇总、交叉检查并写阶段结论

**新增**：`tools/summarize_frequency_followup.py`、`followup/analysis/evidence_matrix.md`。  
**完成标准**：每条假设明确标记 `supported / not_supported / inconclusive`，给出直接 CSV、代码版本、统计单位和边界；最后只做一个 Go / No-go 决策。

### 任务 9｜只有满足门槛才设计下一版

**条件**：P1–P4 至少能形成“新频率信息 → 更优动作选择 → 终态 Size / 同预算”的证据链。  
**若满足**：再设计一个**最小改动**的新模块，并在同数据、同训练预算下与状态 MLP 和普通消息模型进行短训；通过后再做50epoch与1000图正式测试。  
**若不满足**：停止当前分频方向，不把 PAVE 的后处理提升挪作频域贡献。

---

## 12. 容易踩的坑：工程与科研边界清单

1. **不把 `N×20` 当图像平面**。20 是轨迹维，没有二维空间邻接或二维傅里叶意义。
2. **能量邻接边权 0.5 与二值图算子不能混用**。`state_bands()` 已使用 `topology.binary`，不要再无意乘以 2。
3. **有符号高频 `B0/B1` 不能直接取绝对值作为网络信号**。只有在描述性频谱能量里才使用平方；改变符号会改变决策信息。
4. **`Z=L2+B0+B1` 只证明输入分解可逆**。不保证非线性 MLP 后仍能还原、也不等于三部分谱正交。
5. **high 分支输出低频 ≠ high 没用**。图谱中的常数/驻留成分、偏置、SiLU 与 LN 都会影响比例；必须看去驻留后的信息和对决策的实际作用。
6. **低频/高频分量可能有强相关，不能把两个 sigmoid gate 的均值解释成百分比权重**；独立 gate 之和无须为 1。
7. **禁用 low/high 的运行不是“无该模块的重训练消融”**。它只回答现有参数下依赖哪个分支。
8. **当前禁用分支仍计算 MLP**，不能报告“关闭 high 带来速度提升”，除非真实跳过计算并复核输出。
9. **同一 RNG seed 不等于不同计算路径消费了相同随机数**。最好明确预生成 `x0` 和 `U`，再逐元素传给各模型。
10. **浮点精度会改变 Bernoulli 决策路径**。FP32 与 FP16 分别比较，不能把跨精度 Mean Size 差解释成纯数值精度损失。
11. **不要只取期望能量作为结果**。最终 MIS 论文主指标是解码后的 Mean Size↑，速度 Time↓，输出合法性必须检查。
12. **训练目标与动作收益不同**。现有 erdoes 是一步期望能量＋翻转数正则；即便 P3 不佳，也不等于目前算法实现错误。
13. **冻结 PAVE 既是工程要求也是归因边界**：若改动 PAVE，整体收益无法归因于频域结构。
14. **已重复用于模型选择的验证集不能成为新的独立统计证明**。诊断结果要标明 exploratory/confirmatory 的角色，正式测试集只在锁定新方法后使用。
15. **不要复用文件名覆盖历史结果**。`outputs/v4_frequency` 作为历史完整实验，新增 followup 目录只追加。
16. **不要把 graph/node/action/replicate 统计单位混成一层**。图级对照的方差要按图和训练种子处理。
17. **候选采样偏置必须声明**。分层样本的正收益占比不是所有节点上的天然正收益比例。
18. **不要未经证据引入复杂正交、对比、因果或多任务损失**。在 v4 已无独立收益的前提下，引入新损失会显著增加解释成本。

---

## 13. 推荐阅读与代码溯源（本轮以仓库证据为主）

### 必读源码与报告

- [v4 频域模型 `mis_v4_frequency.py`](https://github.com/YuqiWang-code/RLNN_lab/blob/013cbcbf1a55e77548abf07d4cdf656e99c097a3/models/mis_v4_frequency.py)
- [v4 频谱观察器 `diagnose_v4_frequency.py`](https://github.com/YuqiWang-code/RLNN_lab/blob/013cbcbf1a55e77548abf07d4cdf656e99c097a3/models/diagnose_v4_frequency.py)
- [v4 成对收益探针 `probe_v4_utility.py`](https://github.com/YuqiWang-code/RLNN_lab/blob/013cbcbf1a55e77548abf07d4cdf656e99c097a3/models/probe_v4_utility.py)
- [v4 正式验证 `eval_v4_frequency.py`](https://github.com/YuqiWang-code/RLNN_lab/blob/013cbcbf1a55e77548abf07d4cdf656e99c097a3/models/eval_v4_frequency.py)
- [RLNN 求解器与训练 `solvers.py`](https://github.com/YuqiWang-code/RLNN_lab/blob/013cbcbf1a55e77548abf07d4cdf656e99c097a3/models/solvers.py)
- [v4 实验运行说明](https://github.com/YuqiWang-code/RLNN_lab/blob/013cbcbf1a55e77548abf07d4cdf656e99c097a3/train_scripts/v4_frequency/README.md)
- [v4 结果与机制分析](https://github.com/YuqiWang-code/RLNN_lab/blob/013cbcbf1a55e77548abf07d4cdf656e99c097a3/docs/temporary/RLNN_v4_Frequency_Results_Analysis.md)
- [v3 保留的 PAVE H1 与独立验证成绩](https://github.com/YuqiWang-code/RLNN_lab/blob/013cbcbf1a55e77548abf07d4cdf656e99c097a3/docs/temporary/RLNN_v3_Results_Analysis.md)

### 2024–2026 方法背景（已在项目文档调研过；这里仅列与诊断机制直接相关的）

| 工作 | 为什么与这轮实验有关 | 论文 PDF | GitHub |
|---|---|---|---|
| GRACE · NeurIPS 2025 | 低／高专家与节点条件 gate 已有先例；不能把本轮 gate 当作天然创新 | [PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/eb154abcbfa13d23f01ad3e091e92b08-Paper-Conference.pdf) | [代码](https://github.com/KEAML-JLU/GRACE) |
| WaveGC · ICML 2025 | 多尺度谱小波与融合；提醒输入滤波与输出谱特征并非同义 | [PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/liu25y/liu25y.pdf) | [代码](https://github.com/liun-online/WaveGC) |
| PolyGCL · ICLR 2024 | 可学习谱滤波响应；先验证其任务价值再考虑引入 | [PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/6faf3b8ed0df532c14d0fc009e451b6d-Paper-Conference.pdf) | [代码](https://github.com/ChenJY-Count/PolyGCL) |
| GCON / 跨组合优化 · ICML 2026 | 图多滤波器已经用于 MIS，独立创新应在动态搜索决策价值 | [PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/canturk26a/canturk26a.pdf) | [代码](https://github.com/semihcanturk/COPT-MT) |
| HeroFilter · NeurIPS 2025 | 自适应频率选择；不要将普遍异配直觉直接套到 MIS | [PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/5d570ed1708bbe19cb60f7a7aff60575-Paper-Conference.pdf) | [代码](https://github.com/zshuai8/HeroFilter) |

**本文件不包含这些论文的新实验或复现。** 其作用是解释要把“分频作用”与“额外参数、普通邻域聚合、状态特征、额外时间”严格区分。

---

## 14. 一页实施清单与最终交付要求

- [ ] 记录仓库 SHA、三种子权重 hash、数据 manifest 与 PAVE hash；确认未写测试集。
- [ ] 通过 `check_adapter.py` CPU/CUDA 自检、新的逐层跟踪一致性与固定 RNG replay 自检。
- [ ] 补充 `frequency_diag_utils.py`，复核 raw 与去驻留后的谱比例。
- [ ] 扩展 `diagnose_v4_frequency.py` 保存每阶段 `Linear/SiLU/Linear/LayerNorm` 的频谱。
- [ ] 开展 P1：20 图探索 → 三种子复核，回答“哪层产生低频主导”。
- [ ] 新增 P2 同状态干预与相同 `x0,U` 全轨迹 replay，比较 Full/NoLow/NoHigh/FixedGate/NoAdapter 与 B/MLP/MSG。
- [ ] 改造 `probe_v4_utility.py`，支持 R=2/4/8、Uniform/Stratified，写逐重复收益。
- [ ] 先评价标签可靠性，再在共享标签上测 F4 是否降低动作 regret。
- [ ] 新增独占 GPU 的 P4 profiler，拆分真实 E2E 耗时与逐算子微基准。
- [ ] 逐图配对、按图聚合、按种子计算 Std，生成四张表、五张图、evidence_matrix。
- [ ] 结果写 `followup/analysis/decision.json`，明确 `continue / stop / inconclusive`。
- [ ] PAVE-Stats H1 全程冻结；**未确认新频域独立收益前不启动50轮重训练、不运行1000图正式测试、不设计v4-B。**

**本轮完成的标志不是“出现一个更大的 Size 数字”，而是形成可追溯证据链：**

> **图信号 → 滤波与编码 → 门控和 logits → 真实动作收益 → 完整轨迹合法解质量 → 总求解时间。**

任何一环缺乏证据，应标为 `inconclusive`，而不是用“高频更重要”“低频冗余”“模型过拟合”等未经排除其他原因的说法代替分析。

---

**文档修订说明**：GitHub v4 已完成的实验、当前代码签名及命令以 `013cbcb` 为准；本文件所有 P1–P4 扩展、代码片段和 CLI 契约是**建议实现**。模型权重、GPU 服务端图数据和新诊断结果不包含在此文件内。实际编写新增代码后，先自检、后冒烟、再放大统计，并在新提交上更新 SHA 与结果链接。
