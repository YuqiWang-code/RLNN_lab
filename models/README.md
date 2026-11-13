# RLNN 源码

上游：[Shengyu-Feng/RLD4CO](https://github.com/Shengyu-Feng/RLD4CO)，作者 Shengyu Feng、Yiming Yang。基于项目根目录提供的 ZIP，版本与适配记录见[源码来源](../docs/源码来源.md)；[原始 README](README.upstream.md)保留作者说明和引用。包内未提供许可证文件。

当前研究入口是 **RLNN / erdoes**。RLSA 是退火方法，`reinforce` 是另一种训练损失，均保留上游能力，但不是本项目默认基线。

本目录按"是否还在跑"分三组：**现行**（v8 宏步与定位/步数）、**冻结协议**（所有人共用的评测与 PAVE）、**已关闭线**（保留可复现，不再改）。新增模块请放到对应组里，不要在目录顶部平铺。

## 现行：v8 宏步（ACMD）与定位/步数

**宏步线停在 P0**（两个 P0 诊断都无最终质量信号，见 `../docs/temporary/RLNN_v8_P0_Results.md`），
目录里现在真正在跑的是**定位与步数**：等预算对照 KaMIS、把冻结步捕获成 CUDA 图块提高步数。

| 文件 | 职责 |
| --- | --- |
| `mis_v8_macro.py` | 宏步核心：`MacroKernel`（读出 1→2M，第一路精确拷贝骨干）、`widen_readout`、`sample_macro`（每粒子每步抽一个潜分量）、`teacher_blocks`（教师轨迹切块与 `Y`/`B` 标签）、`joint_log_likelihood`、`marginal_bce`（A6 对照） |
| `v8_collect.py` | 冻结教师切块落盘：`Y=x_h`、`B=argmin_{1..h}E`，写 `outputs/v8_macro/data/blocks_{split}_ref{r}.pt` |
| `v8_train.py` | A2/A3/A4/A5/A6 五臂共用一份损失实现，只训补偿头，骨干冻结 |
| `v8_eval.py` | `fixed_nfe` 与 `budget` 两口径；冻结 PAVE 一律 `horizon=1`，档案只收截止前已完成的合法解 |
| `check_v8.py` | M=1 读出等值、联合似然退化、块记录语义三组自检（不需要检查点） |
| `diagnose_v8_prior.py` | P0-B 相关性打乱 + P0-C 端点档案漏失；只读冻结策略与冻结 PAVE，不训练 |
| `mis_v8_capture.py` | `CapturedSearch`：把 10 步冻结步捕获成一块 CUDA 图并重放，`register_generator_state` 保证每次重放的随机数与 eager 一致；`improvements` 计数在图上，`skip_unchanged` 据此跳过重复解码（可证明等价） |
| `check_capture_budget.py` | 捕获闸门：同种子下断言 state / 档案状态 / 档案能量 / 解码掩码与 eager 逐位相等，再断言跳过解码不改主解，然后才计时（2.16 → 0.27 ms/步） |
| `mis_v8_decode.py` | `CapturedDecoder`：把 A.3 解码的迭代循环搬进捕获块（`chunk=4` 最快，1.09 ms vs 6.69 ms），`decode_best` 只在设备端取最大粒子、单列回主机 |
| `check_decode_capture.py` | 解码闸门：4 图 × 4 时间点 × 4 个 chunk，先断言位集与 `solution_masks` 相同再报速度 |
| `probe_decode_cost.py` | 解码成本分解：A.3 迭代数、单迭代 / 稀疏 mm / 单次 sync / 捕获重放，用来定位「解码受限」而非算术受限 |
| `profile_budget_step.py` | 每步成本分解：eager / captured / forward / energy / decode，用来判断瓶颈是算术还是发射延迟 |
| `eval_kamis_anytime.py` | KaMIS（ReduMIS）四档 `--time_limit` anytime 曲线，按图配对到 v7 冻结基线 |
| `eval_v8_capture_budget.py` | 等时钟对照：`--arms`（引擎）× `--decoders`（解码）的格子，同种子、同解码节奏、冻结 PAVE `horizon=1`，每格都与 `eager/eager` 按图配对 |
| `summarize_v8_positioning.py` | 把 KaMIS、两份预算 CSV、解码消融合成「墙钟 vs 解」定位表 |
| `eval_pave_external.py` | 把 PAVE 接在外部求解器输出上：`--source kamis\|min_degree`，六个臂（原始 / 一次扩展 / PAVE 学习·规则·随机 / 长 KaMIS），每臂强制独立集与单调性断言 |
| `eval_regime.py` | 跨族/跨规模迁移：`--families RB_small_val ER_200_300 BA_200_300 RB_large`，三臂（`rl` 捕获搜索@1.0 s / `kamis` 整预算给算法时间 / 贪心+PAVE），每族 40 图、按图配对自助 |
| `summarize_v8_claim.py` | 主张分解表：用 CSV 里的 `search_size`（平台前）与 `final_size`（平台后）算「捕获 × 平台」2×2，判断两个组件是相加还是替代 |
| `merge_v8_shards.py` | 合并 `--shards N` 产生的分片 CSV（按 `graph_id/budget/cell/seed/repeat` 去重，`graph_id` 用全局下标） |

结果见 [`../docs/temporary/RLNN_v8_Positioning_Results.md`](../docs/temporary/RLNN_v8_Positioning_Results.md)，
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
