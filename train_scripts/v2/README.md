# v2 MIS 实验

SEGF 在原 RLNN 的 `Linear(1,128)` 后加入残差适配器。低频使用原图惰性随机游走的一阶、二阶输出，高频使用相邻尺度差及状态条件交换图的残差。每条粒子单独计算选中邻居数和可释放节点。交换图传播把两端掩码分解到同一次稀疏邻居求和，空邻域用恒等传播。两路分别经 32 维 MLP、LayerNorm 和独立门控，残差系数可学习，初值为 0.01。原 Bernoulli 搜索和 erdoes 目标沿用，最后对每条轨迹的最低能量状态做 A.3 解码。

| 组别 | `--frequency` | 适配器输入 |
| --- | --- | --- |
| B | `none` | v1 原 RLNN 权重 |
| B-tuned | `none` | 原 RLNN 同预算全模型微调 |
| F1 | `segf_mlp` | 原始状态，适配器参数量与 F8 相差 61 |
| F2 | `segf_stats` | v1 的四项 MIS 状态统计 |
| F3 | `segf_static1` | 静态一阶低频和差分 |
| F4 | `segf_static2` | 静态双尺度低频和差分 |
| F5 | `segf_exchange` | 状态统计、释放计数和唯一阻挡节点的释放规模 |
| F6 | `segf_high` | 静态双尺度差分、交换图差分 |
| F7 | `segf_fixed` | 双分支固定 0.5 权重 |
| F8 | `segf` | 双分支独立 Sigmoid 门控 |

训练使用 1,000 张 RB 图、100 张验证图、FP32、每批 8 图 × 64 状态、K=20 和 50 步采样轨迹。F1–F8 从同种子 RLNN 的 `best.pt` 初始化，第 1 轮只训练适配器（Adam，1e-4），第 2–3 轮解冻至 2e-5。B-tuned 同数据、同更新数和学习率调度。B、B-tuned 和 F8 使用种子 0、1、2，其余适配器用种子 0。每轮用固定随机流和 A.3 选择验证最优 checkpoint。只有预训练适配器在三个验证种子上对原 RLNN 不降、平均超过同预算微调对照，才开 50 轮的从头训练。

## 合法交换

每轮枚举最多 48 个交换，最多执行 3 轮。1→2 检查两个新节点是否兼容及其唯一阻挡节点。扩展对照在没有 1→2 时枚举 2→3，检查三个新节点之间及与保留节点的约束。更新后用最小度顺序补点并验证整个独立集。

PAVE 使用对加入节点顺序对称的局部特征和 32 维排序 MLP。候选共享同一解，后续两轮使用同一覆盖度规则，训练损失只用最终 Size 不同的候选对。推理用 PAVE 选第一步，后续与标签的继续规则相同。验证集先跑规则、覆盖度启发式、随机顺序和 2→3 后备。种子 0 的 100 图中至少 50 图有 1→2 才使用单一 1→2 邻域，否则使用 2→3 后备。所用邻域至少在 5 张验证图上有非同分候选且累计至少 10 个非同分候选对，才训练正式排序头。

## 测试和记录

测试使用 RLD4CO 作者的 1,000 张 RB200–300 图，顺序及摘要见 `outputs/v2/dataset_manifest.json`。可用 `models/data/prepare_mis_tensors.py` 生成 `models/data/MIS_tensors/RB_small_author_test`，测试入口按原文件名还原顺序。固定步数用 FP16、K=20、T=100、搜索种子 1017 加图序号。同一个模型和图的后处理对照共享已解码的起始解。求解时间从数据上卡后的同步点开始，覆盖拓扑构造、搜索、解码、交换和可行性检查。固定总预算为 0.5 秒，交换组在搜索阶段预留 20 ms，搜索每 10 步检查已完成解；交换只提交截止前通过检查的解。单次 GPU 操作跨过截止时，保留上一份合法解并记录实际总时间和超时数。

每组输出 `run.json` 和 `per_graph.csv`。表中 ΔSize、胜平负按同种子、同预算的 B 配对，另记录 1–500、501–1,000 两组用时。三个种子的汇总用样本标准差。机制诊断记录五个轨迹阶段的交换边、兼容释放对、频带区分能力、门控直方图、残差比例和合法解大小曲线。对不可行原始状态统计局部兼容释放对，对合法状态另计实际可执行交换。

```bash
export PYTHON=/home/hzeng/envs/rlnn/bin/python
bash train_scripts/v2/diagnose_exchange.sh
bash train_scripts/v2/sweep.sh train 0 & bash train_scripts/v2/sweep.sh train 1 & wait
bash train_scripts/v2/sweep.sh test 0 & bash train_scripts/v2/sweep.sh test 1 & wait
```

单组训练使用 `SEED=0 CUDA_VISIBLE_DEVICES=0 bash train_scripts/v2/train_frequency.sh segf`。从头训练用 `FROM_SCRATCH=1`。交换排序训练用 `train_value.sh`，2→3 后备用 `--larger`。独立评估用 `eval.sh` 指定 `--save_dir`、`--frequency`、`--label`、`--post` 和 `--result_dir`。在 Windows 端用 `python tools/publish_v1.py --version v2 --watch` 同步并发布已完成的结果。
