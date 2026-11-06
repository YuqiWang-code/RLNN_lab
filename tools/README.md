# 实验工具

命令从项目根目录执行。服务器连接使用 Python 和 `paramiko`，快照工具只依赖 Python 标准库。

## 同步与状态

```powershell
python -m pip install -r tools/requirements.txt
python tools/server.py status
python tools/server.py status --save tmp/server_status.txt
python tools/server.py push
python tools/server.py pull
```

连接参数读取 `.vscode/sftp.json`。`push` 上传 Git 已登记文件的当前内容，以大小和修改时间跳过未变化文件；同步后保留修改时间。`outputs/` 由服务器产生，使用 `pull` 下载。同步均不删除目标端文件。删除或重命名源码后需要同时整理服务器对应旧文件。

下载范围为 `outputs/` 下 `.log/.csv/.json/.jsonl/.txt/.md/.yaml/.yml`，单个文件不超过 50 MiB，跳过 `checkpoints/`、`wandb/` 和 `__pycache__/`。大型文件在服务器路径中查看。

## 数值整理

```powershell
node tools/collect_results.mjs
```

Node 脚本使用本机 Codex 随附的 `@oai/artifact-tool`，位置为用户目录下 `.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/`。此命令在本机执行。

输入为 `outputs/<版本>/<运行名>/` 下的 `run.json` 和 `metrics/metrics.csv`。训练入口自动生成这些文件。扫描全部 CSV 中的数值，空值不计为 0，非有限值会报告源文件和指标。整理前先 `pull`，保留 CSV 原件。

工作簿 `docs/实验结果.xlsx` 的 `Results` 工作表每行是一项原始指标观测：

| 字段 | 含义 |
| --- | --- |
| `version`, `run` | 输出目录中的版本与运行名 |
| `task`, `dataset`, `split` | 任务、数据路径、train/val/test 划分 |
| `epoch`, `step` | CSV 原始 epoch 与 step |
| `metric`, `value`, `unit` | 原始指标名、数值、单位 |
| `source`, `recorded_at` | 来源 CSV 相对路径、该文件修改时间（UTC） |
| `notes` | 实验备注 |

gap 保存源码输出的百分数，例如 `1.25` 表示 `1.25%`。time 单位为秒；名称带 `_MB` 的显存指标由源码除以 `1024**2` 得到，因此登记为 MiB。损失、目标值和比例沿用原值。每次更新完整替换同一来源 CSV 的记录，其他来源保留；对应行的手工备注随来源刷新，因此长期实验备注保存在运行目录的 Markdown 文件中。

可指定 `--outputs`、`--workbook`，或用 `--preview tmp/results.png` 生成预览。没有实测结果时只创建表头。

## 版本快照

```powershell
python tools/snapshot.py --version v000_baseline
```

输出 `docs/temporary/<版本>_<时间>.txt`，包含：

- 导出时间、Git 基准提交和相关工作区是否有修改。
- 当前 `models/` 源文件及 `train_scripts/` 启动脚本。
- 当前工作簿的全部工作表，以制表符分隔保存单元格值，公式同时保存表达式和已有值。

快照省略数据集、图片、编译产物、缓存和检查点。新版本使用新的版本名，先整理数值，再生成快照。快照用于阅读与归档，源码恢复以 Git 为准。
