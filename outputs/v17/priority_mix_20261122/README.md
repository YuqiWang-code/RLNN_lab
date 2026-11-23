# 排序权重与排序一致性（v17，2026-11-22）

在 200 ms 截止下，用同一批实验重测可行搜索中「学习分数」的权重，回答两个问题：

1. 把混合权重 λ 从 0.1 变到 1.0、或完全去掉度数项，交付规模会变多少？
2. 学习分数本身与度数排序有多接近？

## 运行配置

- 求解器：`models/eval_priority_search.py`，检查点 `outputs/v11/scale_priority_20261121/training/best.pt`（与论文主表同一份），
  调度与论文一致：小型 RB 先团分支 ≤185 ms，大型 RB 先交换再团分支 175 ms，ER 全程交换，3 ms 返回余量。
- 数据：作者公开测试图，`RB_small` 1,000、`RB_large` 1,000、`ER_small` 128、`ER_large` 16；种子 1017/1019/1021。
- 臂：`degree`（λ=0）、`learned_0.1`、`learned`（λ=0.25，实际配置）、`learned_1.0`、`neural`（只用学习分数）。
  五个臂在同一批次、同一张 RTX 5090 上串行运行，交付即计分，未交付按 0；每图三种子取平均后再按图重采样 5,000 次。

## 结果（交付规模，200 ms）

| 混合方式 | RB-[200–300] | RB-[800–1200] | ER-[700–800] | ER-[9000–11000] |
|---|---:|---:|---:|---:|
| 仅度数 | 20.1150 | 42.4387 | 44.2500 | 365.6250 |
| λ=0.1 | 20.1150 | 42.4573 | 44.1432 | 364.3958 |
| λ=0.25（实际配置） | 20.1150 | 42.4933 | 44.4245 | 365.7083 |
| λ=1 | 20.1150 | 42.4950 | 44.2005 | 365.5833 |
| 仅学习 | 20.1150 | 42.5130 | 44.3828 | 364.3750 |

相对同一批度数臂的配对差（95% 区间，`summary/summary.md`）：

- 大型 RB：+0.0187 [+0.0007,+0.0357]（0.1）、+0.0547 [+0.0370,+0.0727]（0.25）、+0.0563 [+0.0380,+0.0743]（1.0）、+0.0743 [+0.0557,+0.0923]（仅学习）。
- 小型 ER：−0.1068 [−0.1901,−0.0234]（0.1）、+0.1745 [+0.0911,+0.2552]（0.25）、−0.0495 [−0.1328,+0.0339]（1.0）、+0.1328 [+0.0443,+0.2266]（仅学习）。
- 大型 ER（16 图）：−1.2292 [−2.9375,+0.2500]（0.1）、+0.0833 [−0.9792,+1.2083]（0.25）、−0.0417 [−0.8333,+0.8333]（1.0）、−1.2500 [−3.0417,+0.2500]（仅学习）。
- 小型 RB：所有臂都达到 20.1150（全部已证明最优）。

排序一致性（`ranks/*/summary.json`，每族 200 个图，小 ER 128 个）：

| 数据集 | ρ(学习分数, −d) | ρ(混合排序, −d) | ρ(混合, 学习) | 前 5% 重叠（学习 vs 度数） |
|---|---:|---:|---:|---:|
| RB_small | 0.5829 | 0.9903 | 0.6512 | 0.1290 |
| RB_large | 0.6451 | 0.9906 | 0.7102 | 0.1805 |
| ER_small | 0.9562 | 0.9986 | 0.9693 | 0.7187 |

结论：学习项的作用随图族变化，量级在 0.02–0.17 个顶点；实际使用的 λ=0.25 在小型 ER 上最好，
大型 RB 上更大的权重（含只用学习分数）还能再多 0.02–0.03 个顶点；混合排序与度数排序的相关性高达 0.99，
说明它主要在度数排名接近的顶点之间重排。论文正文 4.2 一句概述，附录 A.8 给出表与讨论。

## 文件

- `<family>/<arm>/per_graph.csv`：逐图逐种子记录（final_size、wall_ms、final_ready_ms、steps、prepare_ms、score_ms 等）。
- `summary/summary.csv|paired.csv|summary.md|summary.json`：按图配对的均值与区间。
- `ranks/<dataset>/per_graph.csv|summary.json`：排序相关性诊断。

## 复现

```bash
CUDA_VISIBLE_DEVICES=0 bash train_scripts/v17/priority_ranks_ablation.sh outputs/v17/priority_mix_20261122 \
  learned:0.1 learned:1.0 neural
CUDA_VISIBLE_DEVICES=0 bash train_scripts/v17/priority_ranks_ablation.sh outputs/v17/priority_mix_20261122 \
  degree learned
python tools/summarize_priority_mix.py --root outputs/v17/priority_mix_20261122 \
  --reference degree --output outputs/v17/priority_mix_20261122/summary
python tools/analyze_priority_ranks.py --dataset models/data/MIS_tensors/RB_small_author_test \
  --checkpoint outputs/v11/scale_priority_20261121/training/best.pt \
  --output outputs/v17/priority_mix_20261122/ranks/RB_small_author_test
```
