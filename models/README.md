# RLNN 源码

上游：[Shengyu-Feng/RLD4CO](https://github.com/Shengyu-Feng/RLD4CO)，作者 Shengyu Feng、Yiming Yang。基于项目根目录提供的 ZIP，版本与适配记录见[源码来源](../docs/源码来源.md)；[原始 README](README.upstream.md)保留作者说明和引用。包内未提供许可证文件。

当前研究入口是 **RLNN / erdoes**。RLSA 是退火方法，`reinforce` 是另一种训练损失，均保留上游能力，但不是本项目默认基线。

本目录按"是否还在跑"分三组：**现行**（v8 宏步）、**冻结协议**（所有人共用的评测与 PAVE）、**已关闭线**（保留可复现，不再改）。新增模块请放到对应组里，不要在目录顶部平铺。

## 现行：v8 宏步（ACMD）

| 文件 | 职责 |
| --- | --- |
| `mis_v8_macro.py` | 宏步核心：`MacroKernel`（读出 1→2M，第一路精确拷贝骨干）、`widen_readout`、`sample_macro`（每粒子每步抽一个潜分量）、`teacher_blocks`（教师轨迹切块与 `Y`/`B` 标签）、`joint_log_likelihood`、`marginal_bce`（A6 对照） |
| `v8_collect.py` | 冻结教师切块落盘：`Y=x_h`、`B=argmin_{1..h}E`，写 `outputs/v8_macro/data/blocks_{split}_ref{r}.pt` |
| `v8_train.py` | A2/A3/A4/A5/A6 五臂共用一份损失实现，只训补偿头，骨干冻结 |
| `v8_eval.py` | `fixed_nfe` 与 `budget` 两口径；冻结 PAVE 一律 `horizon=1`，档案只收截止前已完成的合法解 |
| `check_v8.py` | M=1 读出等值、联合似然退化、块记录语义三组自检（不需要检查点） |
| `diagnose_v8_prior.py` | P0-B 相关性打乱 + P0-C 端点档案漏失；只读冻结策略与冻结 PAVE，不训练 |

启动器见 [`../train_scripts/v8_macro/`](../train_scripts/v8_macro/README.md)。

## 冻结协议：骨干、搜索与 PAVE

这些文件被所有版本共用，改动会同时污染历史基线与新结论；除修 bug 外不要改，修了要在结果文档里注明。

| 文件 | 职责 |
| --- | --- |
| `main.py` | CLI、种子与求解器选择；`build_parser()` 供项目启动器复用 |
| `models.py` | GCN、MLP 与图卷积（论文数学实现，逐行对应） |
| `solvers.py` | RLNN/RLSA 训练、采样、评价；`flip_logits`/`sample_states`/`policy_topology` 是各版本共用的取数口 |
| `mis_v1.py` | `adjacency`、`search`（轨迹与档案）、`best_decoded`（A.3） |
| `utils.py` / `problems/` / `data/` | 数据与目标函数；MIS 缓存由 `data/prepare_mis_tensors.py` 创建并保留原始图顺序 |
| `mis_v3_search.py` / `mis_v3_value.py` | 合法平台搜索与 PAVE；v4/v7/v8 沿用同一冻结实现 |
| `eval_v1.py` / `eval_v2.py` / `eval_v3.py` | 三代评测器；`eval_v2.dataset_at`/`load_policy`/`write_run` 与 `eval_v3.build_parser` 是现行脚本的公共入口 |
| `collect_v3_starts.py` / `diagnose_v3.py` / `train_exchange_value_v3.py` | PAVE 训练与诊断（历史产物 H1 由它们产生） |

`eval_v3` 默认 `horizon=1`；任何绕过它直接调 `platform_search` 的诊断脚本都必须显式传 `horizon=1`，否则会偏离部署协议（v7 的 `diagnose_v7_census` 曾用 2，是已记录的偏差）。

## 已关闭线：保留可复现，不再修改

| 文件 | 线 | 结论 |
| --- | --- | --- |
| `mis_static_prior.py` / `eval_static_prior*.py` / `train_static_direction.py` / `diagnose_static_leverage.py` / `probe_direction_agreement.py` / `static_prior_protocol.py` | 静态先验（一次性图编码 + 双动作偏置） | 无可复现增益；详见 [structure/README.md](../docs/已归档的想法和文件/structure/README.md) |
| `mis_structure.py` / `train_structure.py` / `eval_structure.py` | 结构补偿（度数/门控/增益） | v1 +0.090 判为单种子波动，v2 `degree_out` +0.013 不显著 |
| `archived_frequency/` | 频域适配器 | 已归档，模型/训练/机制诊断全部移入该子目录 |
| `mis_v7_context.py` / `v7_structure.py` / `train_v7_context.py` / `eval_v7_context.py` | v7 上下文/静态编码 | 方向学不出来，余弦 0.088 |
| `mis_v7_recombine.py` / `mis_v7_conflict.py` / `mis_v7_fast.py` / `diagnose_v7_census.py` / `diagnose_v7_context.py` / `eval_v7_tccr.py` / `eval_v7_budget.py` / `check_v7.py` / `probe_v7_context.py` / `profile_v7_context.py` / `v7_runs.py` | v7 TCCR（精确重组 + 冲突释放） | 固定工作量 +0.088 但预算门四格 CI 全跨零，`budget_fail`；见[结果文档](../docs/temporary/RLNN_v7_TCCR_Results.md) |

`v7_runs.py` 虽是 v7 命名的，但 `backbone`/`pave`/`digest`/`provenance`/`preflight_policy` 是 v8 仍在用的公共工具，不要跟着 v7 一起清掉。

## 目录

| 路径 | 内容 |
| --- | --- |
| `problems/` | MIS、最大团、最大割的数据与目标 |
| `data/` | RB/ER/BA 图生成源码及本地数据位置 |
| `scripts/` | 作者原始脚本，参数差异见项目基线说明 |
| `logs_train/` | 作者日志，仅供参考，不导入实验台账 |
| `environment.yml` | 作者原始环境快照，供溯源 |

从本目录运行 `python main.py --help`；正式项目入口在 `../train_scripts/legacy/v000_rlnn_baseline/`。启动器进入本目录，`from models import GCN` 加载本目录的 `models.py`。不使用 `python -m models.main`。

工程适配包括参数解析器、可配置多图合批和外围结果工具。多图路径对每图损失取平均，并并行生成独立轨迹；默认网络、能量函数及解码保留原 RLNN 行为。RTX 5090 环境见[服务器说明](../docs/服务器.md)。`--resume` 接收 epoch 边界训练状态；当前多图训练每轮保存 `latest_training.pt`，包含 Adam 和主进程随机状态。
