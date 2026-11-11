# 结构补偿（冻结）

本目录是已归档研究线的脚本，保留可复现性，不再迭代。归档总结（三版结论、证据链、失败原因、可复用资产）见 [docs/已归档的想法和文件/structure/README.md](../../docs/已归档的想法和文件/structure/README.md)。

v1 研究点：**在冻结骨干内部补偿消息传递对结构信号与组合增益的压缩**，而不是在输出端加静态偏置。v2 把度数通道搬到读出端。

RLNN 的骨干是 `GCN(1, 128, 5, 1)`——唯一输入通道是二值状态，度、邻居占用、翻转增益全部要靠归一化后的平均聚合重建。高度数节点恰恰是 MIS 最需要精细判断的对象（冲突多、释放空间大），也恰是最容易被平均掉的一类。本目录迭代三个零初始化的补偿通道：

| 臂 | 机制 | 参数 | 位置 |
| --- | --- | ---: | --- |
| `degree` | 每层残差流追加 `W_l · φ(deg)`，`φ` 为归一化对数度的 Fourier 编码（2 + 2×3 维） | 5 × 10 × 128 | 残差流 |
| `gate` | 每层把聚合消息按度数缩放：`m_i ← m_i · (1 + tanh(g_l)·(u_i − mean u))` | 5 + 5 | 聚合输出 |
| `gain` | 显式组合增益残差注入翻转 logits：选中邻居数、未覆盖邻居数、占用比、解析局部增益、归一化度 → MLP，标量门 `tanh(γ)` | 145 + 1 | logits |
| `all` | 三者同时开启 | 785 | — |

全部零初始化，因此**第 0 步逐位等于 B**，`train_structure.py` 的 `compensation_effect` 会用"同检查点清零补偿"给出精确的效应量（对应 v1 的 masked 控制）。

## 协议

- 骨干冻结（`outputs/v2/none_seed{0,1,2}`），只训练 `compensation.*`；教师在冻结 B 的轨迹上生成，Fp32、K20、`num_tp=50`。
- 训练 3 epoch × 2000 updates（1,000 图 × graph batch 8 × state batch 64），默认 `lr=5e-4`、`lambd=0.5`（与 B 自己的 Erdős 训练同口径）。
- 评测沿用 `eval_v3`：验证集前 100 图、A.3 解码、冻结 PAVE-Stats H1、K20/T100 与 1000、以及 0.5/1.0 秒两档预算。
- 主口径 FP32 ＋ CUDA Graph；正式计时每卡只跑一个进程。

## 准入

`summarize.py` 对每个非 `baseline` 臂给出四项：

| 门 | 条件 |
| --- | --- |
| `fixed` | T100 ＋ PAVE 的图级配对 ΔSize 95% CI 下界 > 0 |
| `speed` | T100 平均求解时间 ≤ B 的 1.08× |
| `budget` | 0.5 与 1.0 秒两档的 CI 下界都 > 0 |
| `feasible` | 全部可行、无冲突 |

四项全过记 `advance`，否则 `no_go`。任一臂 `advance` 才进入三种子与作者 1,000 图。

## 运行

```bash
bash train_scripts/structure/check.sh                 # 语法 + 零初始化恒等 + 批不变量
CUDA_VISIBLE_DEVICES=0 bash train_scripts/structure/train_all.sh
CUDA_VISIBLE_DEVICES=0 bash train_scripts/structure/train.sh degree --gpu 0
CUDA_VISIBLE_DEVICES=0 bash train_scripts/structure/eval.sh fixed baseline --gpu 0
CUDA_VISIBLE_DEVICES=1 CUDA_GRAPH=1 bash train_scripts/structure/eval.sh fixed gate --gpu 1
BUDGET=0.5 bash train_scripts/structure/eval.sh budget all --gpu 0
python train_scripts/structure/summarize.py --ref 0
```

`REFERENCE_SEED=0/1/2` 分别对应骨干权重与训练种子 42/123/999。输出写入 `outputs/structure_v1/`，汇总与判决写入 `outputs/structure_v1/analysis/`。

## 设计依据

见 [结构补偿 v1 方案](../../docs/已归档的想法和文件/structure/RLNN_Structure_v1_Design.md) 与[文献索引](../../docs/参考文献/文献索引.md)的"结构补偿"一节。

## 判决

**No-go（v1 与 v2 都是）。** `fixed` 与 `speed` 两门各自独立失败，`budget` 未跑；v2 用每图 3 个搜索种子把 v1 的 +0.090 复现为 +0.0133。阈值、读数与本轮未做项见归档总结。
