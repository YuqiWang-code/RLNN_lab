# 历史研究方案

本目录保存 v1–v3 的研究综述与结构设计。当前频域研究方案见 [v4 重设计](../已归档的想法和文件/RLNN_Frequency_v4_Literature_Code_Redesign.md)，冻结的 PAVE 实验依据见 [v3 结果分析](../temporary/RLNN_v3_Results_Analysis.md)。各版本源代码、训练入口和实验结果保留在原位置。

| 文件 | 内容 |
| --- | --- |
| 当前想法.md | 初期实验设想 |
| RLNN_2024_2026_Frequency_Causal_Literature_Review.md | v1 文献与三个创新方向 |
| RLNN_v2_Frequency_ActionValue_Redesign.md | SEGF 与动作价值方案 |
| RLNN_v3_Exchange_Search_Design.md | 合法中性平台搜索与 PAVE 训练设计 |

## 已归档研究线

| 目录 | 研究点 | 结论 |
| --- | --- | --- |
| [mamba/](mamba/README.md) | 一次性静态结构先验（PC-OSM-RLNN → 静态先验 v1 → v2 接口普查与方向学习） | **No-go**。P0 证明注入接口有杠杆（oracle T100 +0.45、0.5 秒 +0.19），但三次方向学习都失败，瓶颈在 8 维静态特征的输入信息（可预测上限 AUC ≈0.64 → T100 ΔSize +0.003） |

归档目录内的文档保留原文；跨目录链接已按新位置改写。对应代码冻结在 `models/` 与 `train_scripts/mamba/`，实验记录保留在 `outputs/v7_context/`、`outputs/static_prior_v1/`、`outputs/mamba_v2/`。
