# RLNN Lab

RLNN Lab 基于 Shengyu Feng 和 Yiming Yang 的 RLD4CO 实现，研究 RB200–300 图上的最大独立集（MIS）。RLNN 负责搜索初始解，PAVE-Stats H1 在合法解上做平台搜索。[RLD4CO 官方仓库](https://github.com/Shengyu-Feng/RLD4CO)

## 当前方法与实验结论

**限时部署采用原骨干持续搜索＋冻结 PAVE-Stats H1。** PAVE 的后处理收益来自 v3 的 1,000 张作者测试图。骨干部署选择来自 v5 的 100 张验证图、三种子预算对照，两组结果各按原协议报告。[v3 结果与口径](docs/temporary/RLNN_v3_Results_Analysis.md) · [v5 部署对照](docs/已归档的想法和文件/RLNN_v5_Frequency_Results.md)

| 版本 | 结论 | 结果与入口 |
| --- | --- | --- |
| v3 · PAVE-Stats H1 | 同一批合法起点上，固定工作量的 Mean Size 为 19.6633，相比 rule23 提高 0.0603。 | [分析](docs/temporary/RLNN_v3_Results_Analysis.md) · [运行协议](train_scripts/legacy/v3/README.md) |
| v4 · 双尺度频域适配器 | 短筛未显示频域独立收益。机制追诊把高支路的驻留分量与动作收益分开验证。 | [短筛结果](docs/已归档的想法和文件/RLNN_v4_Frequency_Results_Analysis.md) · [机制追诊](docs/已归档的想法和文件/RLNN_v4_Frequency_Mechanism_Followup_Results.md) · [训练与评测](train_scripts/legacy/v4_frequency/README.md) · [追诊复算](train_scripts/legacy/v4_frequency/followup/README.md) |
| v5 · DAFR-Lite | 112 参数的输出残差达到高支路零驻留目标。FP32 固定 100 步的 Mean Size 为 19.580，MSG 为 19.583，MLP 为 19.587。两档预算也未通过准入。 | [实验报告](docs/已归档的想法和文件/RLNN_v5_Frequency_Results.md) · [阶段决策](outputs/v5_frequency/analysis/decision.json) · [训练与评测](train_scripts/legacy/v5_frequency/README.md) |
| v6 · SC-RLNN | 三种子首轮筛选中，固定 100 步＋PAVE 的 Mean Size 为 19.073，配对 B 为 19.657。速度和质量均未通过准入。 | [模型与首轮结果](docs/已归档的想法和文件/RLNN_v6_SC_RLNN.md) · [训练与评测](train_scripts/legacy/v6/README.md) |

v5 验证中，原骨干持续搜索＋PAVE 在 0.5 秒和 1.0 秒预算下分别取得 **19.677** 和 **19.830**，DAFR＋PAVE 分别为 19.500 和 19.617。[逐种子与配对汇总](outputs/v5_frequency/analysis/) · [实验结果工作簿](docs/实验结果.xlsx)

实验使用两张 RTX 5090。各任务独立单卡运行，训练可同卡并行，正式计时每卡运行一个评测进程。

## RLNN复现基线

从随机初始化训练 50 轮，共 100,000 次 Adam 更新。验证集选出的第 43 轮权重在作者公开测试集上取得 **SIZE 19.549**，比论文表 1 的 19.52 高 0.029。[实验报告](docs/实验报告.md) · [论文表 1](https://arxiv.org/pdf/2502.00277#page=7)

| 权重 | 显卡 | SIZE ↑ | TIME ↓（分钟／500 图） |
| --- | --- | ---: | ---: |
| 从头重训 · 第 43 轮 | RTX 5090 | **19.549** | 4.89 |
| 旧基线 · 第 48 轮 | RTX 5090 | 17.792 | 2.55 |
| 作者公开 erdoes | RTX 5090 | 19.520 | 1.73 |
| 论文表 1 | RTX A6000 | 19.52 | 1.64 |

前三行均使用作者公开的两组各 500 张图和附录 A.3 梯度翻转解码。SIZE 为每图平均独立集大小。TIME 为两组逐图采样与解码总耗时的平均值，各次运行独立计时。[实验结果工作簿](docs/实验结果.xlsx) · [逐图测试记录](outputs/v000_rlnn_baseline/rb_train_author_b64_g8_gpu1_seed0_a3/per_graph.csv)

## 在服务器复跑限时对照

从项目根目录执行。已有验证张量缓存、种子 0 的 v2 RLNN 骨干和 v3 PAVE-Stats H1 权重时，可复跑 0.5 秒的持续搜索对照。

```bash
PYTHON="$HOME/envs/rlnn/bin/python" CUDA_VISIBLE_DEVICES=0 SEED=0 \
  SPLIT=validation SAMPLES=100 BUDGET=0.5 POST=pave \
  RUN_NAME="readme_budget_seed0_$(date +%Y%m%d_%H%M%S)" \
  bash train_scripts/legacy/v5_frequency/eval_budget.sh baseline --extended-search
```

结果写入 `outputs/v5_frequency/<RUN_NAME>/pave/`。完整训练、诊断和双档预算矩阵见 [v5 运行说明](train_scripts/legacy/v5_frequency/README.md)。

## 从头复跑 RLNN 基线

从项目根目录运行。已有训练缓存和作者测试图的部署可直接运行重训入口。

新部署先准备环境和本地生成的 1,000 张训练图、100 张验证图，再生成张量缓存。

```bash
python3.10 -m venv "$HOME/envs/rlnn"
source "$HOME/envs/rlnn/bin/activate"
bash train_scripts/legacy/v000_rlnn_baseline/setup_env.sh
bash train_scripts/legacy/v000_rlnn_baseline/prepare_data.sh
bash train_scripts/legacy/v000_rlnn_baseline/prepare_cache.sh
```

将作者发布的 MIS RB200–300 测试文件 `idx_0.gpickle` 至 `idx_999.gpickle` 放入 `models/data/author_rld4co/MIS/RB_small_test/`。评估按编号分成 0–499 和 500–999 两组。[作者测试图](https://huggingface.co/datasets/shengyuf/RLD4CO/tree/f273683b3fe6faa217de35ead96fbe0ee4a44a4a/MIS/RB_small_test) · [数据集说明](docs/数据集.md)

```bash
PYTHON="$HOME/envs/rlnn/bin/python" bash train_scripts/legacy/v000_rlnn_baseline/retrain_author_erdoes.sh
```

脚本在 GPU 1 上每步合批 8 张图，每图最多 64 个状态，以单卡近似作者公开脚本的 8 卡更新方式，每轮更新 2,000 次。训练结束后，脚本用验证最优权重自动跑两组 A.3 测试。本次采用公开脚本的 batch 64，论文附录 A.2 记录的是每卡 32。[训练配置对照](docs/实验报告.md#按作者脚本从头训练) · [服务器说明](docs/服务器.md)

## 目录与结果整理

| 目录 | 用途 |
| --- | --- |
| `models/` | 上游源码与模型迭代 |
| `train_scripts/legacy/v000_rlnn_baseline/` | 环境、数据、训练、评估和冒烟入口 |
| `outputs/v000_rlnn_baseline/<run>/` | 日志、配置和数值结果，模型权重保存在服务器 |
| `train_scripts/legacy/v3/` | PAVE 起点生成、标签、训练和评测 |
| `train_scripts/legacy/v4_frequency/` | 共享教师状态训练、独占计时验证、频谱与收益诊断 |
| `train_scripts/legacy/v5_frequency/` | DAFR-Lite 短训、固定步数与限时评测、机制诊断 |
| `train_scripts/legacy/v6/` | SC-RLNN 训练、速度准入、固定步数评测与逐层诊断 |
| `train_scripts/mamba/v0/` | 一次性 Mamba 静态先验、预检、同信息对照训练与限时评测 |
| `outputs/v3/`、`outputs/v4_frequency/`、`outputs/v5_frequency/`、`outputs/v6/` | 各版本逐图记录、配对汇总与阶段决策 |
| `reference_methods/` | [参考方法代码与来源](reference_methods/README.md) |
| `docs/` | 服务器、数据集、参考文献与实验结果 |
| `docs/temporary/` | 研究方案、实验报告和版本快照 |
| `docs/已归档的想法和文件/` | [历史研究方案](docs/已归档的想法和文件/README.md) |
| `tools/` | 运行记录、同步、结果整理与快照工具 |
| `tmp/` | 本地临时材料 |

在 Windows 本地拉回 v5 的小型运行记录后，使用提供 `@oai/artifact-tool` 的 Node 运行时更新 v5 工作表，并生成代码和数值结果的快照。训练图和模型权重保存在服务器。[工具说明](tools/README.md) · [v5 TXT 快照](docs/已归档的想法和文件/v5_frequency_20261109_174609.txt)

```powershell
python tools/server.py pull --version v5_frequency
node tools/update_v5_workbook.mjs
python tools/snapshot.py --version v5_frequency
```

## 参考文献

Shengyu Feng 和 Yiming Yang，*Regularized Langevin Dynamics for Combinatorial Optimization*，ICML 2025，PMLR 267，16977–16992。[正式论文页](https://proceedings.mlr.press/v267/feng25r.html) · [本地 PDF](docs/参考文献/神经组合优化/RLNN%28ICML2025%29.pdf) · [文献索引](docs/参考文献/文献索引.md) · [源码来源](docs/源码来源.md)
