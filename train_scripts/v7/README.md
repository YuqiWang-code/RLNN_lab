# v7 · TCCR 普查

研究点：**利用轨迹信息定位传统邻域触达不到的联合改进**（TCCR：Trajectory-guided Conflict-Certified Reoptimization）。本目录只放第一阶段的**不训练普查**：先确认这两类联合动作在 RB200–300 上到底还有多少可拿的量，再决定要不要做模型。

方案与门槛：预注册见 [docs/temporary/RLNN_v7_TCCR_Prereg.md](../../docs/temporary/RLNN_v7_TCCR_Prereg.md)。本目录的脚本只跑普查，不训练、不改 PAVE、不写回搜索。

| 文件 | 作用 |
| --- | --- |
| `common.sh` | 路径、Python、`OUT_V7`、`SEEDS`、`backbone_dir`、`--gpu` 解析 |
| `check.sh` | 算子单测（`models/check_v7.py`）＋ 捕获一致性（自己的 T100 捕获 vs `eval_v2.search`） |
| `probe.sh` | 单参考种子普查，`recombine` / `conflict` / `both` 三选一 |
| `probe_pipeline.sh` | 端到端固定工作量：A.3 / 重组 / 冻结 PAVE / 重组+PAVE 四个解的大小 |
| `probe_budget.sh` | 截止时间两臂（同预算、重组从搜索里扣时间）＋ B1–B5 判决 |
| `probe_all.sh` | 多参考种子循环 + 汇总判决 |
| `summarize.py` | 图级配对统计、bootstrap CI、G1–G5 门槛与 `decision.json` |
| `summarize_budget.py` | ABBA 重复平均、图级配对 Δ、B1–B5 门槛与 `decision_budget.json` |

## 运行

```bash
CUDA_VISIBLE_DEVICES=0 bash train_scripts/v7/check.sh
CUDA_VISIBLE_DEVICES=0 bash train_scripts/v7/probe.sh both
REFERENCE_SEED=0 bash train_scripts/v7/probe_pipeline.sh
BUDGETS="0.5 1.0" bash train_scripts/v7/probe_budget.sh
REFERENCE_SEED=1 bash train_scripts/v7/probe.sh both --gpu 1
REFS="0 1 2" bash train_scripts/v7/probe_all.sh both
```

`SAMPLES=100`、`SEEDS="1017 1019 1021"` 为默认；`CUDA_GRAPH=1` 可切到 CUDA Graph 捕获路径（与 `eval_structure` 同口径）。`REFERENCE_SEED=0/1/2` 对应骨干与 PAVE 的种子 0/1/2。

## 产物

| 路径 | 内容 |
| --- | --- |
| `outputs/v7_tccr/census/recombine_ref{r}.csv` | 每（图, 搜索种子）一行：去重数、两两 Hamming/Jaccard、190 对 Oracle、四个固定预算选对策略、每次重组耗时 |
| `outputs/v7_tccr/census/conflict_ref{r}.csv` | 每（图, 搜索种子, 层）一行：`a3` / `pave` 两层的区域数、改善区域数、`k=2/3/4` 的最大增益、窗口大小、普查耗时 |
| `outputs/v7_tccr/pipeline/pipeline_ref{r}.csv` | 每（图, 搜索种子）一行：A.3／重组／PAVE／重组+PAVE 四个解的大小与分段耗时 |
| `outputs/v7_tccr/pipeline/run_ref{r}.json` | 端到端运行的配置、均值与 SHA |
| `outputs/v7_tccr/budget/budget_ref{r}.csv` | 每（图, 种子, 预算, 臂, 重复）一行：解大小、搜索步数、段耗时、是否超时 |
| `outputs/v7_tccr/analysis/summary_budget_ref{r}.json` | 两档预算的配对 Δ、CI 与 B1–B5 判决 |
| `outputs/v7_tccr/census/run_ref{r}.json` | 配置、种子、耗时、骨干与 PAVE 的 SHA、数据清单 SHA |
| `outputs/v7_tccr/checks/run.json` | 捕获一致性逐图记录（`match=1` 表示与 `eval_v2.search` 逐位相同） |
| `outputs/v7_tccr/analysis/decision.json` | G1–G5 与最终判决（`advance` / `recombine_only` / `conflict_only` / `stop`） |

## 口径

- K20 / T100 / A.3 / 冻结 PAVE-Stats H1，验证集前 100 图，非测试集。
- 图级值先在搜索种子内平均，再按图 bootstrap；不把 (图, 种子) 当独立样本。
- 所有合法解都用两套独立实现交叉校验：`SearchGraph.feasible`（搜索侧）与 `mis_v7_recombine.is_independent`（bitset 侧）。
- 局部 MIS 精确解上限 60 节点，超限退化为贪心并在 `exact_k=0` 记录；普查中出现 `exact_k=0` 时结论只看有精确解的行。

## 停止规则

G1–G5 任一组合不满足时按预注册判决表处理，不调门槛、不换图子集、不换种子救回。`stop` 时下一步是诊断 KaMIS 那 0.44 的差距落在哪类图/哪类结构上，而不是继续堆模块。

## 本轮结论

三个参考种子一致：重组在**冻结 PAVE 之上**还有 +0.088（69/231/0），区域通道只有 +0.043（8–15/100 图）。但预算门（0.5 秒／1.0 秒）给出 −0.028／+0.010，20 ms 的搜索时间值 +0.060 > 重组增益 +0.033，按预注册是 `budget_fail`：这条线到此为止。完整读数、机制与下一步见 [docs/temporary/RLNN_v7_TCCR_Results.md](../../docs/temporary/RLNN_v7_TCCR_Results.md)。
