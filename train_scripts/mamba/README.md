# Mamba / 静态先验线（冻结）

本目录是已归档研究线的脚本，保留可复现性，不再迭代。归档总结（三版结论、证据链、失败原因、可复用资产）见 [docs/已归档的想法和文件/mamba/README.md](../../docs/已归档的想法和文件/mamba/README.md)。

| 目录 | 内容 |
| --- | --- |
| `v0/` | PC-OSM-RLNN 预检（正确性、独占计时、跨图代理、重编号诊断）与 P0 脚本 |
| `v1/` | 静态先验 v1（双常量 + 中心化节点项）训练与四协议评测 |
| `v2/` | P0 接口普查（26 臂）、oracle 方向训练、权重与纯监督判决实验、闭包脚本 |

入口仍需 `PYTHON`、`RLNN_SOURCE_SHA` 与 `models/data/MIS_tensors` 缓存；v2 的 `common.sh` 会 `source` v0 的 `common.sh`，因此不能在缺少该链的情况下单独运行。

判定：**No-go，瓶颈在输入信息**（8 维静态标量的可预测上限 AUC ≈0.64 → T100 ΔSize ≈+0.003，低于 0.17 门槛）。后续的"结构补偿"研究点也已在 v2 之后归档，见 [train_scripts/structure/](../structure/README.md)。
