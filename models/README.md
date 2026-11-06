# RLNN 源码

上游：[Shengyu-Feng/RLD4CO](https://github.com/Shengyu-Feng/RLD4CO)，作者 Shengyu Feng、Yiming Yang。基于项目根目录提供的 ZIP，版本与适配记录见[源码来源](../docs/源码来源.md)；[原始 README](README.upstream.md)保留作者说明和引用。包内未提供许可证文件。

当前研究入口是 **RLNN / erdoes**。RLSA 是退火方法，`reinforce` 是另一种训练损失，均保留上游能力，但不是本项目默认基线。

| 文件 | 职责 |
| --- | --- |
| `main.py` | CLI、种子与求解器选择；`build_parser()` 供项目启动器复用 |
| `models.py` | GCN、MLP 与图卷积 |
| `solvers.py` | RLNN/RLSA 训练、采样、评价 |
| `problems/` | MIS、最大团、最大割的数据与目标 |
| `data/` | RB/ER/BA 图生成源码及本地数据位置 |
| `scripts/` | 作者原始脚本，参数差异见项目基线说明 |
| `logs_train/` | 作者日志，仅供参考，不导入实验台账 |
| `environment.yml` | 作者原始环境快照，供溯源 |

从本目录运行 `python main.py --help`；正式项目入口在 `../train_scripts/v000_rlnn_baseline/`。启动器进入本目录，`from models import GCN` 加载本目录的 `models.py`。不使用 `python -m models.main`。

工程适配仅提取参数解析器，添加外围运行、元数据及结果工具，训练和解码逻辑保持原样。RTX 5090 的待验证环境方案见[服务器说明](../docs/服务器.md)，不直接安装原始 CUDA 12.4 环境。
