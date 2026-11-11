# 结构补偿（已归档）

本目录归档 2026-11-10 至 2026-11-11 的第四条研究线：**在冻结 RLNN 骨干内部补偿消息传递对结构信号（度数）与组合增益的压缩**。终局：**No-go，且"度数"这条信号三次注入都只到 +0.01–+0.09，未复现**。部署仍为 B 持续搜索 ＋ 冻结 PAVE-Stats H1。

## 三版结论

| 版本 | 做了什么 | 结论 |
| --- | --- | --- |
| v6 · SC-RLNN | 换骨干：状态感知的自身/邻居和差混合、跨层差分读出、五维输入含度数 | No-go：T100＋PAVE 19.073 vs B 19.657（−0.58），FP16 1.38×/FP32 1.51×；高带 logit 二阶矩 0.0011 < 0.01，第五层能比 0.108–0.133 < 0.25 |
| v1 · 层内三通道 | 保持 B，只加零初始化残差：`degree`（逐层 Fourier 度数残差）、`gate`（度数条件聚合缩放）、`gain`（显式组合增益 MLP） | No-go，但出现首个非零信号：`degree` T100＋PAVE **+0.090**（CI [−0.020, +0.200]，19/70/11）；`gate` +0.060、`gain` +0.030、`all` −0.020；速度 1.11×–1.29× |
| v2 · 读出端融合 | `degree_out` 把 8 维度数特征拼进 `final_project`（零新增 kernel）；`all_out` 加 softplus 正尺度的增益；每图 3 个搜索种子 | No-go：`degree_out` 只有 **+0.0133**（CI [−0.040, +0.063]，20/64/16），`all_out` 与它逐位相同，速度仍 1.11×。**v1 的 +0.090 判为单种子波动** |

## 证据链

三次尝试分别排除了"换网络"、"换注入位置"、"换参化方式"，剩下的是同一件事：这条信息对 B 的决策没有增量。

1. **换骨干不行**（v6）：和差混合 + 跨层读出在短训首轮把裸解从 19.53 打到 17.44，加 PAVE 仍差 0.58；频域健康指标同时不达标。
2. **注入位置不是问题**（v1）：逐层残差、聚合门控、logits 残差三个位置全试过，`degree` 最强也只到 +0.090 且未复现。
3. **参化不是问题**（v2）：`gain` 换成可学正尺度 + 零初始化末层后，输出与不含 `gain` 的 `degree_out` **逐位相同**——残差本身就是 0，问题在梯度量级而非门幅。
4. **代价不可接受**（v1/v2）：单看算子，读出端一次广播加已经是最小形态，仍要 +11%。基线每步是十几个小 kernel 的发射受限状态，**每步多一个算子就按它的发射成本计费**。
5. **机制现象一致**：补偿把 logits 推动 2.1–2.5（RMS），翻转概率只动约 1.2%。与已归档的静态先验同源——B 的 logit 分布大面积饱和，只有少数边界节点对任何加性偏置敏感。

## 文档

| 文件 | 内容 |
| --- | --- |
| [RLNN_Structure_v1_Design.md](RLNN_Structure_v1_Design.md) | v1 设计：三个补偿通道、零初始化恒等、实验矩阵、四条判读规则、预留的 oracle 上界臂、S01–S17 文献对应 |
| [RLNN_Structure_v1_Results.md](RLNN_Structure_v1_Results.md) | v1 结果：四臂 fixed 读数、`comp_delta_rms` vs 概率改变的落差、准入判决、v2 计划 |
| [RLNN_Structure_v2_Results.md](RLNN_Structure_v2_Results.md) | v2 结果：读出端融合、三种子、+0.0133、`all_out` 逐位相同、三条结论与"未做"清单 |
| [../RLNN_v6_SC_RLNN.md](../RLNN_v6_SC_RLNN.md) | v6 SC-RLNN 的模型、五组对照、首轮结果与适用范围（前一次结构尝试） |

## 代码（冻结，不删除）

- `models/`：`mis_structure.py`（`STRUCTURE_MODES` / `StructureCompensation` / `StructureTopology`）、`train_structure.py`、`eval_structure.py`，以及 `solvers.py`、`main.py`、`capture_mis.py` 里的 `structure_mode` 接线。
- `train_scripts/structure/`：`check.sh`（零初始化恒等 + 开关扰动 + 批不变量 + 门界 + 缓存一致性）、`train_all.sh`、`eval.sh`、`eval_matrix.sh`、`summarize.py`。
- `outputs/structure_v1/`、`outputs/structure_v2/`：逐图 CSV、`summary_ref*.json`、`decision_ref*.json` 保留作证据。

## 可复用的资产

1. **零初始化 + 同检查点清零**这套控制：任何往冻结骨干里加通道的点，都能用它把"没学到"和"没用"分开，并且第 0 步逐位等于基线（`check.sh` 直接对 `torch.equal` 断言）。
2. **发射受限基线的成本口径**：B 每步十几个小 kernel，新增一个逐 step kernel ≈ +0.2 ms ≈ +11%。任何"每步多一个算子"的设计可以先按这个价目表判死。
3. **`comp_delta_rms` / `prob_mean_abs` 双读数**：前者说明通道有没有被用上，后者说明有没有可能改变决策。二者差两个数量级就是这个基线里加性偏置的典型命运。
4. **别用单种子下结论**：+0.090 在 n=100、每图 1 种子时 SE≈0.05，本身就是 1.8σ；v2 用 3 个搜索种子后只剩 +0.013。后续任何臂默认三种子起。

## 未做（不影响结论）

`budget` 0.5/1.0 秒两档；ref1/2 复制种子；用 3 个搜索种子正式复测 v1 的 `degree` 臂（把 +0.090 判为波动）；作者 1,000 图。前三项若要写进论文才需要补，`speed` 与 `fixed` 两门已各自独立失败，补跑不改变判决。
