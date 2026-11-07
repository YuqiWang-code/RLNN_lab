# RLNN Lab

v1 在 RB200–300 MIS 上独立比较解码质量感知的候选保存、动态高低频协同、成对动作收益及其组合。[v1 实验协议](train_scripts/v1/README.md) · [参考实现](reference_methods/README.md) · [实验结果](docs/实验结果.xlsx)

基于 Shengyu Feng 和 Yiming Yang 的 RLD4CO 实现，研究最大独立集（MIS）的神经组合优化方法。当前基线为 `v000_rlnn_baseline`，任务为 RB200–300，训练和评估使用单张 RTX 5090（GPU 1）。[RLD4CO 官方仓库](https://github.com/Shengyu-Feng/RLD4CO)

## 当前基线

从随机初始化训练 50 轮，共 100,000 次 Adam 更新。验证集选出的第 43 轮权重在作者公开测试集上取得 **SIZE 19.549**，比论文表 1 的 19.52 高 0.029。[实验报告](docs/实验报告.md) · [论文表 1](https://arxiv.org/pdf/2502.00277#page=7)

| 权重 | 显卡 | SIZE ↑ | TIME ↓（分钟／500 图） |
| --- | --- | ---: | ---: |
| 从头重训 · 第 43 轮 | RTX 5090 | **19.549** | 4.89 |
| 旧基线 · 第 48 轮 | RTX 5090 | 17.792 | 2.55 |
| 作者公开 erdoes | RTX 5090 | 19.520 | 1.73 |
| 论文表 1 | RTX A6000 | 19.52 | 1.64 |

前三行均使用作者公开的两组各 500 张图和附录 A.3 梯度翻转解码。SIZE 为每图平均独立集大小。TIME 为两组逐图采样与解码总耗时的平均值，各次运行独立计时。[实验结果工作簿](docs/实验结果.xlsx) · [逐图测试记录](outputs/v000_rlnn_baseline/rb_train_author_b64_g8_gpu1_seed0_a3/per_graph.csv)

## 在服务器复跑

从项目根目录运行。已有训练缓存和作者测试图的部署可直接运行重训入口。

新部署先准备环境和本地生成的 1,000 张训练图、100 张验证图，再生成张量缓存。

```bash
python3.10 -m venv "$HOME/envs/rlnn"
source "$HOME/envs/rlnn/bin/activate"
bash train_scripts/v000_rlnn_baseline/setup_env.sh
bash train_scripts/v000_rlnn_baseline/prepare_data.sh
bash train_scripts/v000_rlnn_baseline/prepare_cache.sh
```

将作者发布的 MIS RB200–300 测试文件 `idx_0.gpickle` 至 `idx_999.gpickle` 放入 `models/data/author_rld4co/MIS/RB_small_test/`。评估按编号分成 0–499 和 500–999 两组。[作者测试图](https://huggingface.co/datasets/shengyuf/RLD4CO/tree/f273683b3fe6faa217de35ead96fbe0ee4a44a4a/MIS/RB_small_test) · [数据集说明](docs/数据集.md)

```bash
PYTHON="$HOME/envs/rlnn/bin/python" bash train_scripts/v000_rlnn_baseline/retrain_author_erdoes.sh
```

脚本在 GPU 1 上每步合批 8 张图，每图最多 64 个状态，以单卡近似作者公开脚本的 8 卡更新方式，每轮更新 2,000 次。训练结束后，脚本用验证最优权重自动跑两组 A.3 测试。本次采用公开脚本的 batch 64，论文附录 A.2 记录的是每卡 32。[训练配置对照](docs/实验报告.md#按作者脚本从头训练) · [服务器说明](docs/服务器.md)

## 目录与结果整理

| 目录 | 用途 |
| --- | --- |
| `models/` | 上游源码与模型迭代 |
| `train_scripts/v000_rlnn_baseline/` | 环境、数据、训练、评估和冒烟入口 |
| `outputs/v000_rlnn_baseline/<run>/` | 日志、配置和数值结果，模型权重保存在服务器 |
| `docs/` | 服务器、数据集、参考文献与实验结果 |
| `docs/temporary/` | 当前想法和版本快照 |
| `tools/` | 运行记录、同步、结果整理与快照工具 |
| `tmp/` | 本地临时材料 |

在 Windows 本地拉回运行记录后，使用提供 `@oai/artifact-tool` 的 Node 运行时整理工作簿明细，并生成代码和数值结果的快照。[工具说明](tools/README.md)

```powershell
python tools/server.py pull
node tools/collect_results.mjs
python tools/snapshot.py --version v000_rlnn_baseline
```

## 参考文献

Shengyu Feng 和 Yiming Yang，*Regularized Langevin Dynamics for Combinatorial Optimization*，ICML 2025，PMLR 267，16977–16992。[正式论文页](https://proceedings.mlr.press/v267/feng25r.html) · [本地 PDF](docs/参考文献/神经组合优化/RLNN%28ICML2025%29.pdf) · [文献索引](docs/参考文献/文献索引.md) · [源码来源](docs/源码来源.md)
