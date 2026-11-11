# RLNN 源码

上游：[Shengyu-Feng/RLD4CO](https://github.com/Shengyu-Feng/RLD4CO)，作者 Shengyu Feng、Yiming Yang。基于项目根目录提供的 ZIP，版本与适配记录见[源码来源](../docs/源码来源.md)；[原始 README](README.upstream.md)保留作者说明和引用。包内未提供许可证文件。

当前研究入口是 **RLNN / erdoes**。RLSA 是退火方法，`reinforce` 是另一种训练损失，均保留上游能力，但不是本项目默认基线。

| 文件 | 职责 |
| --- | --- |
| `main.py` | CLI、种子与求解器选择；`build_parser()` 供项目启动器复用 |
| `models.py` | GCN、MLP 与图卷积 |
| `solvers.py` | RLNN/RLSA 训练、采样、评价 |
| `mis_v3_search.py` / `mis_v3_value.py` | 合法平台搜索与PAVE，v4沿用冻结版本 |
| `mis_v7_context.py` / `v7_structure.py` | 一次性静态图编码、双动作偏置和同信息对照 |
| `train_v7_context.py` / `eval_v7_context.py` | 冻结教师训练、固定步数与同预算 PAVE 对照 |
| `mis_v7_recombine.py` | 两个独立集的二部图差异区最优重组（König / 最大匹配），纯 Python bitset，无 torch 依赖 |
| `mis_v7_conflict.py` | 阻挡集合、释放区域 `R(U)`、局部精确 MIS 与上下界（v7 TCCR 普查） |
| `diagnose_v7_census.py` | v7 普查驱动：K=20 合法解捕获、重组 Oracle、区域 Oracle，只读不训练 |
| `eval_v7_tccr.py` | v7 端到端固定工作量：A.3／重组／冻结 PAVE／重组+PAVE 四个解同进程对照 |
| `check_v7.py` | v7 算子的暴力交叉验证、`polish` 不变量与 3→4 构造例（CPU，无检查点） |
| `mis_structure.py` / `train_structure.py` / `eval_structure.py` | 已归档的结构补偿线（度数/门控/增益通道） |
| `archived_frequency/` | 频域模型、训练和机制诊断 |
| `probe_v4_utility.py` | 验证集上共享随机数的真实翻转收益探针 |
| `problems/` | MIS、最大团、最大割的数据与目标 |
| `data/` | RB/ER/BA 图生成源码及本地数据位置 |
| `scripts/` | 作者原始脚本，参数差异见项目基线说明 |
| `logs_train/` | 作者日志，仅供参考，不导入实验台账 |
| `environment.yml` | 作者原始环境快照，供溯源 |

从本目录运行 `python main.py --help`；正式项目入口在 `../train_scripts/legacy/v000_rlnn_baseline/`。启动器进入本目录，`from models import GCN` 加载本目录的 `models.py`。不使用 `python -m models.main`。

工程适配包括参数解析器、可配置多图合批和外围结果工具。多图路径对每图损失取平均，并并行生成独立轨迹；默认网络、能量函数及解码保留原RLNN行为。RTX 5090环境见[服务器说明](../docs/服务器.md)。v4入口与开关见[实验协议](../train_scripts/legacy/v4_frequency/README.md)，`--frequency v4a` 启用频域适配器，默认 `none`。

MIS 缓存由 data/prepare_mis_tensors.py 创建，保留原始图顺序。--resume 接收 epoch 边界训练状态；当前多图训练每轮保存 latest_training.pt，包含 Adam 和主进程随机状态。
