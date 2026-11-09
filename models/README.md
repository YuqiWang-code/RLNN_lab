# RLNN 源码

上游：[Shengyu-Feng/RLD4CO](https://github.com/Shengyu-Feng/RLD4CO)，作者 Shengyu Feng、Yiming Yang。基于项目根目录提供的 ZIP，版本与适配记录见[源码来源](../docs/源码来源.md)；[原始 README](README.upstream.md)保留作者说明和引用。包内未提供许可证文件。

当前研究入口是 **RLNN / erdoes**。RLSA 是退火方法，`reinforce` 是另一种训练损失，均保留上游能力，但不是本项目默认基线。

| 文件 | 职责 |
| --- | --- |
| `main.py` | CLI、种子与求解器选择；`build_parser()` 供项目启动器复用 |
| `models.py` | GCN、MLP 与图卷积 |
| `solvers.py` | RLNN/RLSA 训练、采样、评价 |
| `mis_v3_search.py` / `mis_v3_value.py` | 合法平台搜索与PAVE，v4沿用冻结版本 |
| `mis_v4_frequency.py` | 双尺度频域适配器、同参数MLP/消息对照与推理干预 |
| `train_v4_frequency.py` | 匹配教师状态下的适配器训练 |
| `eval_v4_frequency.py` / `diagnose_v4_frequency.py` | 冻结PAVE评测与频谱/门控诊断 |
| `probe_v4_utility.py` | 验证集上共享随机数的真实翻转收益探针 |
| `problems/` | MIS、最大团、最大割的数据与目标 |
| `data/` | RB/ER/BA 图生成源码及本地数据位置 |
| `scripts/` | 作者原始脚本，参数差异见项目基线说明 |
| `logs_train/` | 作者日志，仅供参考，不导入实验台账 |
| `environment.yml` | 作者原始环境快照，供溯源 |

从本目录运行 `python main.py --help`；正式项目入口在 `../train_scripts/v000_rlnn_baseline/`。启动器进入本目录，`from models import GCN` 加载本目录的 `models.py`。不使用 `python -m models.main`。

工程适配包括参数解析器、可配置多图合批和外围结果工具。多图路径对每图损失取平均，并并行生成独立轨迹；默认网络、能量函数及解码保留原RLNN行为。RTX 5090环境见[服务器说明](../docs/服务器.md)。v4入口与开关见[实验协议](../train_scripts/v4_frequency/README.md)，`--frequency v4a` 启用频域适配器，默认 `none`。

MIS 缓存由 data/prepare_mis_tensors.py 创建，保留原始图顺序。--resume 接收 epoch 边界训练状态；当前多图训练每轮保存 latest_training.pt，包含 Adam 和主进程随机状态。
