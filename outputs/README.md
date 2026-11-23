# 实验输出

逐图记录、运行配置与日志是结果依据。正式和冒烟结果分别入簿，数据与权重存放在服务器。 [项目现状](../docs/项目现状.md) · [正式工作簿](../docs/实验结果.xlsx) · [冒烟工作簿](../docs/冒烟结果.xlsx) · [部署与环境](../docs/服务器.md)

| 阶段 | 原始产物入口 | 主要记录 |
| --- | --- | --- |
| 常驻求解与独立图族 | [四预算逐图记录](v16/anytime_author/) · [ER 同核交错复测](v16/er_interleaved/) · [DIMACS 逐图结果](v16/dimacs_register/by_graph.csv) · [团覆盖证书](v16/dimacs_register/clique_covers.json) | 学习排序、度排序和常驻 OnlineMIS 的主机可行解、截止覆盖与实际返回时间 |
| 重训 RLNN 与可行解搜索 | [三档逐图结果](v15/priority_200ms_20261121/test/) · [正式配对表](v15/priority_200ms_20261121/formal_comparison.csv) · [训练权重和日志](v11/scale_priority_20261121/training/) · [小 RB 逐图最优值](v15/priority_200ms_20261121/offline_optimality/RB_small/optima.csv) | 同一训练权重、学习/纯度/无排序三臂、逐次按时节点集和上界；另含独立离线 SAT 核验与限时 OnlineMIS 对照 |
| 排序权重与排序一致性（v17） | [逐图记录与统计](v17/priority_mix_20261122/) · [排序相关性诊断](v17/priority_mix_20261122/ranks/) | 同一批 200 ms 实验中的度数、λ=0.1/0.25/1.0、仅学习五臂，按图配对区间；学习分数与度数的 Spearman 相关及前 5% 重叠。配置见 [README](v17/priority_mix_20261122/README.md) |
| ICML 论文 | [论文 PDF](paper_20261120/paper.pdf) · [中文阅读版](paper_20261120/paper_zh.pdf) · [Overleaf 源码包](paper_20261120/overleaf-source.zip) · [正文与图表](../overleaf/) | 标题 *SPRINT: Feasible-Set-First Neural Search for Maximum Independent Set Under a 200-Millisecond Deadline*；求解器族 SPRINT 的构造路线 SPRINT-C 与投机路线 SPRINT-S。200 ms 主表、图级配对区间、系统消融、解码轨迹与冷链派车算例 |
| 四档作者测试图的 200 ms 对照 | [四档总表](v11/scale_deadline_20261120/comparison.csv) · [正式工作簿「四档规模 200ms」](../docs/实验结果.xlsx) · [新增三档逐图记录](v11/scale_deadline_20261120/) · [配对区间](v11/scale_deadline_20261120/paired.json) · [原图编号映射](v11/scale_author_data_20261120/) | 统一使用种子 1017、1019、1021。SIZE 含截止前无解的零分，平均墙钟包含超时返回 |
| 运行流程、粒子数与权重 | [逐图结果](v11/paper_ablations/checkpoint_bridge_20261120/) · [相邻配置配对](v11/paper_ablations/checkpoint_bridge_20261120/summary.json) | 作者权重分别用 K=20 和 K=200，比对原协议与相同融合入口的本地权重 |
| 首个解与末段 40 ms | [逐图记录](v11/paper_ablations/decode_reserve_20261120/) · [配对区间与首解阶段耗时](v11/paper_ablations/decode_reserve_20261120/summary.json) | 初始化后前导解码、留空 40 ms、PAVE 或继续搜索 |
| OnlineMIS 六种子复测 | [新增逐图结果与合并统计](v11/online_panel_20261120/) | 原三种子、新三种子在相同的 1000 张图上各配对；合并统计在每张图内平均六次差值 |
| 解码宽度与定性实例 | [宽度配对](v11/paper_ablations/decode_width_20261120/summary.json) · [解码轨迹与节点集](v11/paper_ablations/qual_instance_20261120/case.json) | 同预算对照 top 4 与全部 200 个粒子；在单独诊断中记录首张图每十步的 36 个合法集合 |
| INTEL 与 PPO-LwD 200 ms | [INTEL](v11/intel_ppo_deadline_20261120/intel/) · [PPO-LwD](v11/intel_ppo_deadline_20261120/ppo_lwd/) · [全部限时对照表](v11/intel_ppo_deadline_20261120/comparison.csv) · [全部配对统计](v11/intel_ppo_deadline_20261120/summary.json) | INTEL 发布权重的 GPU 移植，官方 PPO 在 RB 上重训。逐图指标、带可用时间的合法节点集、日志和 PPO 验证选权 |
| DIFUSCO 与 DGL-TreeSearch 200 ms | [DIFUSCO](v11/supervised_deadline_20261119/difusco/) · [DGL-TreeSearch](v11/supervised_deadline_20261119/dgl_treesearch/) · [配对统计](v11/supervised_deadline_20261119/summary.json) | DIFUSCO 官方 SATLIB 权重迁移，DGL-TreeSearch 官方实现重训。含逐图指标、带可用时间的合法集、运行日志和 DGL 训练权重 |
| DiffUCO 与 LTFT 200 ms | [DiffUCO](v11/learned_deadline_20261119/diffuco/) · [LTFT](v11/learned_deadline_20261119/ltft/) · [配对统计](v11/learned_deadline_20261119/summary.json) | DiffUCO 官方发布权重，LTFT 官方实现重训。含逐图指标、带可用时间的合法集、运行日志及 LTFT 训练日志和权重 |
| 作者 RLNN 与 FLD-EG 200 ms | [作者 RLNN](v11/reference_deadline_20261119_followup/author_rlnn/) · [FLD-EG 重建版](v11/reference_deadline_20261119_followup/fld_eg/) · [配对统计](v11/reference_deadline_20261119_followup/summary.json) | 同图同种子逐图指标、解可用时刻、合法节点集和日志 |
| 外部基线 200 ms | [五组运行](v11/reference_deadline_20261119/) · [配对统计](v11/reference_deadline_20261119/summary.json) · [作者节点 ID 映射](v11/reference_deadline_20261119/graph_nodes.jsonl) | 官方 DISCS iSCO 两种停止条件、RLSA、KaMIS ReduMIS 与 OnlineMIS。保留逐图指标、带时间戳的合法解和日志 |
| FP16 与 FP32 | [12 组运行](v11/paper_ablations/precision_20261119/) · [130 步](v11/paper_ablations/precision_20261119/fixed130/summary.json) · [200 ms](v11/paper_ablations/precision_20261119/deadline200/summary.json) | 两组均保留 dense、CUDA Graph 和策略编译，关闭 TF32 |
| Dense 与 CSR 邻接 | [12 组运行](v11/paper_ablations/adjacency_20261119/) · [130 步](v11/paper_ablations/adjacency_20261119/fixed130/summary.json) · [200 ms](v11/paper_ablations/adjacency_20261119/deadline200/summary.json) | 两组均保留 CUDA Graph、关闭策略编译。另存两种邻接的编译探针日志 |
| 200 ms 解码时机 | [六组运行](v11/paper_ablations/decode_schedule_20261119/) · [配对统计](v11/paper_ablations/decode_schedule_20261119/summary.json) | 中段加末次与仅末次，四项指标按同图同种子比较 |
| CUDA Graph × 编译 2×2 | [24 组运行](v11/paper_ablations/cuda_graph_compile_20261119/) · [配对统计](v11/paper_ablations/cuda_graph_compile_20261119/summary.json) | 固定 130 步与 200 ms 截止各四臂、三种子。每组包含逐图 CSV、配置和日志 |
| v13 · 0.2 秒截止配对 | [六次运行](v11/fused_pave_02s_deadline/) · [PAVE 配对](v11/fused_pave_02s_deadline/summary.json) · [作者 RLNN 对照](v11/fused_pave_02s_deadline/vs_rlnn.json) · [固定步数 RLSA 对照](v11/fused_pave_02s_deadline/vs_rlsa.json) | 两个 PAVE 配置、种子 1017／1019／1021。每次运行包含 `v11_rows.csv`、`padded/padded_ref0.json` 和 `run.log` |
| v13 · 早期融合 | [逐图汇总和配对](v11/collect/v13/) · [1 秒三种子](v11/r2/p05/l3_three_seed.json) | 0.1／0.2 秒调度消融、融合与未融合、数值探针以及 1 秒融合 |
| v11–v12 · A0 与机制 | [A0 闸门](v11/gate0/) · [A0 三种子](v11/t7/summary.md) · [v11 汇总](v11/collect/) · [第二轮](v11/r2/) | 固定步数、时钟阶梯、C4、并行多臂与吞吐诊断。 [闭包汇总](v11/p1_law_1000.json) |
| v8–v10 · 捕获与基线 | [v8_macro](v8_macro/) · [RLSA 逐图基线](v1/rlsa/) · [DCSS 部署](v8_macro/v9_scale_deploy/) · [修复信号普查](v8_macro/repair_signal/) | 捕获、作者集、解码、摊薄、跨族、KaMIS、重训与移动集诊断。 [文档索引](../docs/temporary/README.md) |
| v7 与独立研究线 | [TCCR](v7_tccr/) · [v7 context](v7_context/) · [静态先验](static_prior_v1/) · [Mamba v2](mamba_v2/) · [结构 v1](structure_v1/) · [结构 v2](structure_v2/) | 固定工作量、时钟预算、方向探针和准入结果 |
| v1–v6 | [v1](v1/) · [v2](v2/) · [v3](v3/) · [v4](v4_frequency/) · [v5](v5_frequency/) · [v6](v6/) | 合法交换、PAVE、频域和换骨干对照 |
| v000 | [基线运行](v000_rlnn_baseline/) | 环境、数据清单、冒烟、单卡训练与作者集复现。每个运行的 `run.json` 记录类型与状态 |

## 0.2 秒逐图字段

用 `graph_id`、`seed` 和 `repeat` 对齐两臂。`search_size` 是截止前搜索解，`final_size` 是截止时计分的结果。`decode_ms` 包含搜索内解码时间，`last_decode_ready_ms` 记录末次解码的完成时刻。`search_ready_ms` 和 `final_ready_ms` 记录计分解可用时刻，`wall_ms` 和 `deadline_overrun` 记录原始运行墙钟。覆盖率按计分解的可用时刻计算，墙钟超时按原始运行结束时刻计算。 [复跑脚本](../train_scripts/v11/fused_pave_02s.sh)
