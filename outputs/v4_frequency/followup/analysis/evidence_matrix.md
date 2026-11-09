# v4 频率机制证据矩阵

阶段决策为 **No-go**。保留原骨干与冻结 PAVE，当前 F4 结构停在机制筛选。

| 假设 | 判断 | 证据与适用范围 |
|---|---|---|
| H-F1 编码损失了可用高频信息 | not_supported | F4 的 97.111% 驻留能量解释了原始低频占比，扣除驻留后较高频仍占 90.009%。偏置干预不改变对应线性层的非驻留能量，任务相关信息是否可读仍需动作标签检验。[P1](stage_summary.csv) |
| H-F2 双支路抵消或错误竞争 | inconclusive | FP32 同状态的一步能量贡献近似相加，关闭高支路的 Size 差为 0，固定门控下降 0.0700。FP16 关闭任一路均上升，但双路交互的区间跨零。[回放](replay_paired.csv) · [原生重复](native_repeat_summary.csv) |
| H-F3 关键动作上的频率优势被平均掩盖 | not_supported | 分层候选确有局部信号，整体 regret 改善 +0.00069，只有 4/60 图—种子对非零，关键类别未显示一致方向。[收益配对](utility_paired.csv) |
| H-F4 额外开销压缩限时搜索 | supported | F4 的训练图独占 A.3 端到端时间为 709.38 ms，原骨干为 376.78 ms。既有 0.5 秒验证分别搜索 59.09 与 98.12 步。开销分布在统计、传播、编码、门控融合，固定步数的质量差仍在。[独占计时](profile_e2e.csv) · [组件](profile_components.csv) · [限时验证](historical_quality_time.csv) |
| H-F5 固定教师与学生轨迹错位 | not_supported | 切换状态来源后 F4 高支路去驻留比例变动小于 0.05 个百分点，学生状态上的冻结 B 续跑收益也未建立频率优势。[轨迹频谱](stage_by_seed.csv) · [收益配对](utility_paired.csv) |

| 门槛 | 判断 | 依据 |
|---|---|---|
| D1 可靠动作标签 | inconclusive | H=20、R=16 的两半 Spearman 为 0.1523，共同非平局对的同向率为 56.05%。改变 H 未取得稳定排序。[可靠性](reliability_summary.csv) |
| D2 频率独立优势 | not_supported | Uniform16 的 F4 regret 改善为 −0.00139，Stratified24 为 +0.00069，学生状态复核的方向亦不一致。[收益配对](utility_paired.csv) |
| D3 固定工作量 | not_supported | F4 相对 MSG 的 Size 差为 FP32 −0.0167、FP16 −0.0333。FP16 三个种子均为负，图级区间包含零。[回放](replay_paired.csv) |
| D4 部署预算 | not_supported | 既有同预算验证中，F4＋PAVE 为 19.4800，原骨干＋PAVE 为 19.6233，持续搜索原骨干＋PAVE 为 19.6500。[限时验证](historical_quality_time.csv) |

P1/P2 的图为曾用于选权重的 100 张验证图。P3 在训练图上测冻结 B 的单节点续跑价值，P4 的独占计时覆盖 20 张训练图和一个种子。区间以图 ID 为重采样单位。诊断实现版本为 `e28d326`，权重来源见[清单](../manifest.json)。
