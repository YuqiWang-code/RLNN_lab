# docs/temporary — v8 结果记录

一份文档一条线，按时间从早到晚。读结论请看根 `README.md` 的版本表，这里放细节。

| 文档 | 内容 |
| --- | --- |
| `RLNN_lab_v8_review_and_research.md` | v8 的设计底稿（ChatGPT 起草，已按实际口径放宽） |
| `RLNN_v8_P0_Results.md` | 宏步（ACMD）的 P0 闸门：P0-B 相关性、P0-C 端点档案，两项无信号 → 主线停止 |
| `RLNN_v8_Positioning_Results.md` | 定位：KaMIS anytime 曲线、eager vs captured 等钟对照、解码优化 |
| `RLNN_v8_PAVE_External_Results.md` | PAVE 外接到 KaMIS 解：全零；正对照（贪心起点）+0.710 |
| `RLNN_v8_Regime_Probe.md` | 跨族/跨规模迁移：ER、BA、RB-large 全输，退化随离分布距离单调 |
| `RLNN_v8_Paper_Table.md` | 1000 张作者测试图主表 + 与 RLNN/RLSA/KaMIS 的对齐（含 TIME 口径更正） |
| `RLNN_v8_Amortised_Capture.md` | 捕获摊薄：一次捕获服务整个数据集，0.2 s 双轴赢 RLNN |
| `RLNN_v8_Retrain.md` | 重训三连负（微调中性、贪心起点 −0.5、从零 30 epoch 更差）+ 交换/踢动插循环 |
| `RLNN_v8_Global_Token.md` | 全局 token：中性（判据未过） |
| `RLNN_v8_Exchange_Headroom.md` | 学习型交换头：训练前判死（k≤4 的交换见证 0/60） |
| `RLNN_v8_Learned_Annealing.md` | 学习式退火：训练前判死（固定温度单调变差） |
| `RLNN_v8_Large_Moves.md` | 大移动：球内精确修复、簇重建、策略重填、图缩减，全部零或 +0.02 |
| `v8_paper_snapshot_A.txt` | A 版论文快照（主张、主表、负结果清单、复现命令） |

`指标.txt` 是早期指标草稿。