# 工具

从项目根目录执行。Python 同步依赖：`python -m pip install -r tools/requirements.txt`。运行器使用 RLNN 环境的 Python。结果工作簿由 `@oai/artifact-tool` JS 导出，使用 Codex 提供的 Node 运行时；模块从用户目录 `.cache/codex-runtimes/codex-primary-runtime/dependencies/node` 解析，不依赖项目 node_modules。

## 运行与同步

`run_rlnn.py train|eval|smoke --run NAME` 复用 `models/main.py` 参数；训练脚本设置工作目录与项目默认参数。每次输出 `run.json`、原始日志、成功运行的 `metrics/metrics.csv`。eval 需 `--checkpoint DIR`。运行名不可重复。只允许 MIS/RLNN 和可行解解码；loss 可显式改为 reinforce，但 run.json 会记录，不能归为论文 erdoes 对照。

`server.py status` 查看硬件。push/pull 需要本地配置 `remotePath` 和 `deploymentConfirmed=true`；当前独立 RLNN 目录已启用按需同步。push 包括 Git 跟踪及未忽略的新文件，排除 `.vscode`、outputs、tmp；不删除服务器文件。pull 仅拉取新版本 `outputs/v000_rlnn_baseline` 的小文本文件，不下载权重与图文件。

## 结果与快照

```powershell
node tools/collect_results.mjs
node tools/collect_results.mjs --kind smoke
python tools/snapshot.py --version v000_rlnn_baseline
```

正式单卡默认写入 `docs/实验结果.xlsx`，smoke 单独写入 `docs/冒烟结果.xlsx`，仅在存在真实冒烟记录后使用。收集器只扫描 outputs，读取当前版本、对应 kind、状态 completed 且含 run.json 的 CSV。作者 logs_train 不入账。同一 source 重收时替换该来源的旧记录。

Results 长表字段：version、run、task、dataset（实际划分路径）、split、epoch、step、metric、value、unit、source、recorded_at、notes。step 空缺：原日志没有可靠更新计数。notes 保存 kind、seed、模型/训练/搜索配置、设备、原日志目录、参考值来源。raw_obj 保留负目标，independent_set_size 为其相反数；未引入 gap。val/test time 为每图平均秒数，run/wall_time 为外围总秒数。具体统计范围见[基线](../docs/RLNN基线.md)。

snapshot 将 models（含数据生成源码，排除图数据与作者日志）、项目训练脚本及正式/冒烟工作簿单元格值写入 `docs/temporary/<version>_<time>.txt`，记录 Git 基点与工作区状态。原始 ZIP 和大文件不写入快照。tmp 中不再需要的中间文件应及时清理。

## 完成后归档

`python tools/finish_baseline.py --watch` 等待本版本服务器流程结束，确认 50 epoch 与 1000/100/500 数据完整后，拉取结果、整理两个工作簿、生成正式报告与快照，再同步服务器并提交推送。服务器流程失败时记录状态并停止发布；本地状态位于 `tmp/baseline_completion.json`。
