# v4 频域状态适配器

当前实验只改变 RLNN 搜索状态编码。PAVE 使用 `outputs/v3/stats_h1_seed{0,1,2}/checkpoints/best.pt`，24 维统计、H1、32 个中性候选、48 个严格改善候选、三轮初始规则交换、四轮修复与八轮平台搜索均沿用 v3。

## 模型与对照

动态信号为 `Z=[x,log1p(c)]`，`c=Ax`；图算子为二值无向图上的 `P=(I+D^-1 A)/2`。孤立节点的 P 为恒等。静态拓扑在一次求解中复用，状态描述每步重算。频带为 `L2=P²Z`、`B0=Z-PZ`、`B1=PZ-P²Z`，满足重构 `Z=L2+B0+B1`。

| arm | 两路输入 | 用途 |
| --- | --- | --- |
| `none` | 原 RLNN | 同预算继续微调 |
| `v4a_mlp` | Z；x、logc、logdegree、logr | 状态统计与容量对照 |
| `v4a_message` | P²Z；Z、PZ | 同信息、同参数普通消息对照 |
| `v4a` | L2；B0、B1 | 双尺度频域候选 |
| `v4a_fixed` | 同 v4a，门控固定 0.5 | 门控消融 |

三种适配器使用相同的 2→16→128 与 4→16→128 分支、LayerNorm、六维门控输入与独立 sigmoid，两项可学习残差系数初值 0.01。共同门控含状态、度数、选中邻居数、释放数、裁剪能量差、尺度差强度。MLP 是容量对照；其门控仍能访问共同图统计。Message 输入保留 Z、PZ、P²Z，和频域三项线性等价。

## A1 训练协议

从同种子 v2 `none` 的已微调 RLNN 权重启动。种子为 0、1、2；每组 3 epoch、1000 张训练图、100 张验证图、每批 8 图 × 64 状态、6000 次优化更新、FP32、原 erdoes 主目标、Adam。适配器第 1 epoch 冻结骨干、学习率 1e-4，第 2–3 epoch 解冻并使用 2e-5；`none` 对照第一轮训练整个骨干，其他学习率和更新预算一致。

短期筛选使用冻结的同种子起始 RLNN 生成训练状态。教师每图 20 条轨迹、50 步，三组适配器共享图顺序和采样随机流。模型初始化完成后，将训练随机流重设为 `20261+seed`，消除各结构初始化所消耗随机数的差异。教师不参与梯度；学生仍学习原独立 Bernoulli 的一步期望能量和距离正则。训练状态只来自训练集。`STATE_SOURCE=on_policy` 提供原策略生成状态的训练入口，不能与共享教师状态结果混算。

每轮按验证随机流 1017+图序号、K=20/T=100 和 A.3 选择最优权重。正式筛选再次用同一批验证图记录 FP32 与 FP16 结果，以及加冻结 PAVE 前后的 Size；限时结果统一 FP16。所有权重与数据保存在服务器。

## 阶段决策

先运行 4 arm × 3 seed。主筛选口径为 FP32、固定搜索量、不加 PAVE：F4 相对 Message 和 MLP 都需要平均验证 Size 为正，且各至少两个种子严格改善。效率口径为 FP16、0.5 秒、冻结 PAVE：F4 的平均 Size 需超过固定旧骨干、Message 和 MLP。如果质量优势被额外时间抵消，就停止当前结构。`decide.py` 记录各项逐种子差值与阶段决策。未通过筛选时仍记录低/高/门控置零的机制诊断，但不利用测试集挑选结构。

通过后，F4 与相应 MLP/Message 对照统一进行 50 epoch 三种子训练，再展开 low/high、固定门控、释放统计和能量先验消融。动态任务图 v4-B 仅在静态版本显示独立价值后推进。每个阶段均保留不加 PAVE 与加同一个冻结 PAVE 的两张结果表。

## 执行

```bash
export PYTHON=/home/hzeng/envs/rlnn/bin/python
CUDA_VISIBLE_DEVICES=0 "$PYTHON" train_scripts/v4_frequency/check_adapter.py
CUDA_VISIBLE_DEVICES=0 SEED=0 bash train_scripts/v4_frequency/train.sh v4a
```

`RUN_NAME`、`EPOCHS`、`SEED`、`CHECKPOINT`、`BATCH_SIZE`、`GRAPH_BATCH_SIZE` 可指定实验；改变 batch 后必须在全部主对照统一使用，并报告实际更新数。默认保留 64×8，优先用独立训练进程填充两张 GPU。正式计时阶段每张卡只允许一个评测进程，计时包含构图、滤波、GCN、采样、A.3 与 PAVE。

训练输出为 `outputs/v4_frequency/<arm>_seed<seed>/run.json`、`training.csv` 和服务器上的 `checkpoints/`。逐图评测及诊断也写入 `outputs/v4_frequency/`。汇总使用三个种子均值的样本标准差，逐图配对保留图 ID 与种子，不把同一张图的三次采样当作三张独立图。

`screen.sh` 在两卡各启动六个独立训练；`validate.sh 0/1` 等待训练结束后，各自串行执行固定量和限时验证。强规则基线使用 20 ms 后处理预留，PAVE 使用既定 50 ms。`mechanism.sh` 等两卡计时任务均结束后运行三种子各 20 图频谱/门控诊断、真实翻转收益探针，再用独占卡完成 low/high/gate 的推理干预。分支干预复用已训练权重，不等同于从头训练的结构消融。

`strong_pave.sh` 在干预评测结束后，补充三种子原骨干＋冻结 PAVE 的持续搜索预算对照。仅放开骨干的步数上限，仍在0.5秒总预算内为同一个PAVE保留50 ms，其余PAVE配置保持一致。

## 本轮结果

12组训练、102份方法评测、12组频域诊断和3组收益探针已完成。F4在FP32固定量下为19.53000，普通消息为19.54667；0.5秒下F4＋PAVE为19.48000，原骨干持续搜索＋PAVE为19.65000。当前适配器未通过A1，保留v3 PAVE主线。完整均值、标准差、逐图差值和机制解释见[结果分析](../../docs/temporary/RLNN_v4_Frequency_Results_Analysis.md)。

复算顺序为 `python tools/summarize_v4_mechanism.py`、`python train_scripts/v4_frequency/decide.py`、`python tools/analyze_v4_frequency.py`；工作簿使用提供 `@oai/artifact-tool` 的Node运行时执行 `tools/update_v4_workbook.mjs`。
