# 历史研究方案

本目录保存早期设计、研究线复盘及 [早期实验与文章主张](RLNN_实验历史.md)。冻结 PAVE 的实验依据见 [v3 结果分析](v3/RLNN_v3_Results_Analysis.md)，轨迹重组的两轮准入见 [v7 TCCR 结果](v7_tccr/RLNN_v7_TCCR_Results.md)。现行结论见 [项目现状](../项目现状.md)。

| 文件 | 内容 |
| --- | --- |
| 当前想法.md | 初期实验设想 |
| RLNN_2024_2026_Frequency_Causal_Literature_Review.md | v1 文献与三个创新方向 |
| RLNN_v2_Frequency_ActionValue_Redesign.md | SEGF 与动作价值方案 |
| RLNN_v3_Exchange_Search_Design.md | 合法中性平台搜索与 PAVE 训练设计 |
| RLNN_v6_SC_RLNN.md | SC-RLNN：换骨干的结构补偿（结构补偿线第一版） |

## 已归档研究线

| 目录 | 研究点 | 结论 |
| --- | --- | --- |
| [mamba/](mamba/README.md) | 一次性静态结构先验（PC-OSM-RLNN → 静态先验 v1 → v2 接口普查与方向学习） | **No-go**。P0 证明注入接口有杠杆（oracle T100 +0.45、0.5 秒 +0.19），但三次方向学习都失败，瓶颈在 8 维静态特征的输入信息（可预测上限 AUC ≈0.64 → T100 ΔSize +0.003） |
| [structure/](structure/README.md) | 结构补偿（v6 SC-RLNN → v1 层内三通道 → v2 读出端融合） | **No-go**。度数信号三种形态都只有 +0.01–+0.09 且未复现（v2 三种子 +0.0133，CI 跨零）；增益通道两次都打不开；任何每步新增算子都要 +11% 墙钟（基线发射受限） |

归档源码见 [models/closed](../../models/closed/README.md)，运行记录见 [输出索引](../../outputs/README.md)。归档启动器对应各阶段的历史提交。
