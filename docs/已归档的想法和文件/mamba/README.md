# Mamba / 一次性静态先验（已归档）

本目录归档 2026-11-10 至 2026-11-11 的第三条研究线：**给冻结的 RLNN 骨干加"一次性计算的静态结构先验"**。终局：**No-go，瓶颈在输入信息**。部署仍为 B 持续搜索 ＋ 冻结 PAVE-Stats H1。

## 三版结论

| 版本 | 做了什么 | 结论 |
| --- | --- | --- |
| v0 · PC-OSM-RLNN | 一次性双向 Mamba 编码 8 维静态特征，输出两动作偏置 | P0 No-go：等价优化后宽度 16 仍为 B 的 1.216×/1.249×；跨图代理 AUC 0.6402 相对同信息 MLP/MSG 区间跨零 |
| v1 · 静态先验 | 放弃 Mamba 算子，改用 133 参数 MLP（5 维），冻结 B | No-go：同检查点关闭节点项后 250 个配对大小全同；节点偏置 RMS 只有 0.0026 |
| v2 · P0 接口普查 | 26 个臂：振幅扫描、同 RMS 随机方向、KaMIS/B 反向 oracle | **通过**：oracle 在 T100 +0.45、0.5 秒 +0.19、反向 −0.29，空对照带 0.17 |
| v2 · P1 oracle 方向 | 用 KaMIS 训练解做方向标签，搜索目标训练 | No-go：对齐项只占损失 0.19%，cosine −0.04 |
| v2 · 权重判决 | 对齐项权重 0.03 → 30 → 100 | 仍 No-go：权重提高 3,000 倍，cosine 只到 +0.05 |
| v2 · 纯监督判决 | 频率标签 + 加权 BCE + 按 AUC 选点（178 参数） | No-go：留出 AUC 0.6375（复现代理探针的 0.636），但 T100 ΔSize **+0.0033**，CI [−0.057, +0.063]，胜/平/负 36/227/37 |

## 证据链

四个环节各自独立地被排除，剩下的是输入信息：

1. **接口有杠杆**（P0）：oracle-kamis@0.5 在 T100 得 +0.3967、T800 +0.1367、0.5 秒 +0.1900、1.0 秒 +0.0833；anti-oracle 单调为负（−0.16/−0.29/−0.46）。定价曲线近似线性：`ΔSize(T100) ≈ 0.45 × cosine(方向, oracle)`。
2. **算子不是瓶颈**（v0）：MLP / MSG / Mamba 在代理任务上 0.6362 / 0.6347 / 0.6402。
3. **目标不是瓶颈**（v2 三次）：搜索目标训练 −0.04；加权主导 +0.05；纯监督 0.088。
4. **输入信息是瓶颈**：8 维静态标量对"节点是否属于优质解"的可预测上限约 AUC 0.64，对应 cosine ≈0.1，对应 T100 ≈0～+0.04，低于 0.17 门槛。要过 0.5 秒（空对照带 0.10）需 cosine ≳0.6，按实测斜率约需 AUC 0.85 以上。

## 文档

| 文件 | 内容 |
| --- | --- |
| [RLNN_v7_OneShot_Mamba_Research_and_Implementation_Plan.md](RLNN_v7_OneShot_Mamba_Research_and_Implementation_Plan.md) | v7 总方案：PC-OSM-RLNN 的架构、特征、排序、P0/P1/P2 协议与准入标准 |
| [RLNN_Mamba_v0_P0_Results.md](RLNN_Mamba_v0_P0_Results.md) | v0 预检：正确性、独占耗时、跨图代理 AUC、重编号诊断 |
| [RLNN_Static_Prior_v1_Analysis_and_Plan.md](RLNN_Static_Prior_v1_Analysis_and_Plan.md) | v0 逐图耗时分解与 v1 方案（双常量 + 中心化节点项） |
| [RLNN_Static_Prior_v1_Results.md](RLNN_Static_Prior_v1_Results.md) | v1 结果：四臂 × 四协议，同检查点干预严格为零 |
| [RLNN_Static_Prior_v1_Postmortem_and_v2_Plan.md](RLNN_Static_Prior_v1_Postmortem_and_v2_Plan.md) | v1 复盘（六层缺陷）、v2 设计与实验方案、`models/` 改动清单 |
| [RLNN_Mamba_v2_P0_Results.md](RLNN_Mamba_v2_P0_Results.md) | P0 接口普查：26 臂结果、空对照带定义、四条结论、定价曲线 |
| [RLNN_Mamba_v2_P1_Results.md](RLNN_Mamba_v2_P1_Results.md) | P1 全链条：oracle 方向训练、权重判决、纯监督判决、证据链与归档建议 |

## 代码（冻结，不删除）

代码保留在原位以维持可复现与导入路径：

- `models/`：`mis_v7_context.py`、`v7_structure.py`、`mis_static_prior.py`、`v7_runs.py`、`train_v7_context.py`、`eval_v7_context.py`、`eval_static_prior_v2.py`、`diagnose_static_leverage.py`、`probe_direction_agreement.py`、`train_static_direction.py`、`static_prior_protocol.py`。
- `train_scripts/mamba/v0|v1|v2/`：预检、训练、评测与队列脚本。
- `outputs/v7_context/`、`outputs/static_prior_v1/`、`outputs/mamba_v2/`：逐图记录与判决 JSON 保留作证据。

## 可复用的资产

1. **P0 接口普查方法论**：26 臂（振幅扫描 + 同 RMS 随机方向 + 正/反向 oracle）、空对照带定义（换种子与随机臂的有符号效应 CI 绝对端点最大值）、同随机流配对 —— 任何"往 RLNN 里加一个信号"的点都可以直接套用。
2. **定价曲线**：`ΔSize ≈ 0.45 × cosine`，加上四个协议上 oracle 的绝对上限（+0.45 / +0.14 / +0.19 / +0.08）。以后任何静态先验提案，先用它算上限再决定投入。
3. **`models/probe_direction_agreement.py`**：1 分钟给出"方向质量 → 搜索收益"，λ 插值两端精确复现 P0 数值。
