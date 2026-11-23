# v8–v13 实验文档

[项目现状](../项目现状.md) 汇总现行结论与证据边界。 [实验输出](../../outputs/README.md) · [v1–v7 及独立研究线](../已归档的想法和文件/README.md)

| 阶段 | 结果与实现 | 方案与验证 |
| --- | --- | --- |
| v13 · 融合引擎 | [0.2 秒限时配对](../../outputs/v11/fused_pave_02s_deadline/summary.json) · [早期消融与配对](RLNN_v13_Ablation_Results.md) · [引擎实现](RLNN_v13_FusedEngine.md) | [验收协议](RLNN_v13_Engine_Rewrite_Spec.md) · [早期融合复审](RLNN_v13_Review.md) |
| v12 · 并行预算与后处理 | [轨迹方差](RLNN_v12_R2_E1_Variance.md) · [多臂实测](RLNN_v12_R2_E2_Parallel.md) · [吞吐剖析](RLNN_v12_R2_E3_Throughput.md) · [早期 0.2 秒基线审计](RLNN_v12_R2_Budget02_SOTA.md) | [第二轮实验方案](RLNN_v12_Round2_Directive.md) · [闭包与门控备忘](RLNN_v12_SecondContribution.md) |
| v11 · 机制诊断 | [结果汇总](RLNN_v11_Results.md) · [实现与 A0](RLNN_v11_Implementation.md) · [感受野](RLNN_v11_G1_ReceptiveField.md) · [MATLAB 验证](RLNN_v11_MATLAB_Validation.md) | [分析](RLNN_v11_Analysis.md) · [诊断](RLNN_v11_Diagnosis.md) · [方向与判据](RLNN_v11_Direction_Locked.md) · [复审](RLNN_v11_Review.md) · [文献](RLNN_v11_Literature_2024_2026.md) |
| v9–v10 · 算子与移动集 | [DCSS](RLNN_v9_DCSS_Results.md) · [K 与时钟](RLNN_v9_SOTA_Hunt.md) · [更换移动集](RLNN_v10_Move_Set.md) | [RLSA 固定步数时钟与算子设计](RLNN_v8_SOTA_Direction.md) |
| v8 · 捕获与对照 | [作者集主表](RLNN_v8_Paper_Table.md) · [定位和步数](RLNN_v8_Positioning_Results.md) · [捕获摊薄](RLNN_v8_Amortised_Capture.md) · [外接 PAVE](RLNN_v8_PAVE_External_Results.md) · [跨族及跨规模](RLNN_v8_Regime_Probe.md) | [宏步设计](RLNN_lab_v8_review_and_research.md) · [宏步 P0 结果](RLNN_v8_P0_Results.md) |
| v8 · 改策略与扩邻域 | [重训](RLNN_v8_Retrain.md) · [全局 token](RLNN_v8_Global_Token.md) · [交换余量](RLNN_v8_Exchange_Headroom.md) · [退火](RLNN_v8_Learned_Annealing.md) · [大移动](RLNN_v8_Large_Moves.md) | [早期实验与文章主张](../已归档的想法和文件/RLNN_实验历史.md) |

历史快照包括 [v8 论文口径](v8_paper_snapshot_A.txt)、[v10 论文口径](v10_paper_snapshot_A.txt)、[v11 代码和结果](v11_20261118_20261118_063140.txt)、[v13 代码和结果](v13_20261118_20261118_085837.txt)。 [早期指标草稿](指标.txt)
