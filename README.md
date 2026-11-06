# NExCO Lab

基于 NExCO 的扩散组合优化实验，覆盖 TSP、CVRP 和 MIS。模型说明见 [models/README.md](models/README.md)。

## 项目目录

| 目录 | 内容 |
| --- | --- |
| `models/` | 模型、训练入口、数据加载器、解码与本地搜索代码；所有模型迭代在此进行 |
| `train_scripts/<版本>/` | 对应版本的服务器训练和测试启动脚本 |
| `outputs/<版本>/<运行名>/` | 控制台日志、CSV 指标、运行参数；目录与启动脚本版本对应 |
| `docs/参考文献/` | 按研究类型分目录保存参考材料；统一入口为 [文献索引](docs/参考文献/文献索引.md) |
| `docs/` | [服务器信息](docs/服务器.md)、[数据集信息](docs/数据集.md)、[项目现状](docs/项目现状.md)和 [实验结果工作簿](docs/实验结果.xlsx) |
| `docs/已归档文件/` | 已结束或搁置的想法及结论 |
| `docs/temporary/` | [当前想法](docs/temporary/当前想法.md)、各版本代码和数值结果的 TXT 快照 |
| `tools/` | 服务器同步、结果整理和快照工具，见 [工具说明](tools/README.md) |
| `tmp/` | 可清理的临时文件，不提交 Git |
| `.vscode/` | 本机 SFTP 配置；账号密码只保存在本机配置中 |

## 运行实验

TSP100 当前版本为 `v001_tsp100`，初始代码归档为 `v000_baseline`。在服务器项目根目录运行：

```bash
CUDA_VISIBLE_DEVICES=1 bash train_scripts/v001_tsp100/train_tsp100.sh
CKPT_PATH=/absolute/path/to/trained/checkpoint.ckpt bash train_scripts/v001_tsp100/test_tsp100.sh
```

本版脚本默认使用 GPU 1。可用 `RUN_NAME` 指定运行名，每次独立实验使用不同名称。额外训练参数可追加在脚本命令后。原始版本的其他任务脚本位于 `train_scripts/v000_baseline/`。

运行产物位于 `outputs/v001_tsp100/<运行名>/`：`console.log`、`run.json`、`metrics/metrics.csv`，以及服务器上的检查点和 W&B 离线记录。检查点和大型中间文件留在服务器。一轮训练后，完整 TSP100 测试集的扩展后平均 gap 为 **1.0085%**；配置与结果见 [TSP100 基线](docs/TSP100基线.md)。

## 训练结束后

本机项目根目录执行：

```powershell
python tools/server.py pull
node tools/collect_results.mjs
python tools/snapshot.py --version v001_tsp100
git add .
git commit -m "Record TSP100 baseline experiment results"
git push origin main
python tools/server.py push
```

整理工具读取 CSV 中的全部非空数值，保留 epoch、step、数据划分、单位和来源。重复执行会刷新相同日志的记录。`docs/实验结果.xlsx` 是结果总表；没有实测结果时保留空表。TXT 快照包含当前 `models/` 源码、启动脚本以及工作簿全部工作表的单元格内容。

## 本地与服务器同步

连接配置为 `.vscode/sftp.json`，模板见 [.vscode/sftp.example.json](.vscode/sftp.example.json)。

```powershell
python -m pip install -r tools/requirements.txt
python tools/server.py status
git add .
python tools/server.py push
```

`push` 上传 Git 已登记的工作区文件，不删除服务器文件；修改后尚未提交的已登记文件也会上传。`pull` 下载 `outputs/` 下的日志、指标和参数文本，跳过检查点、W&B 二进制目录及超过 50 MiB 的单个文件。新建源码先 `git add` 再同步。

GitHub：[YuqiWang-code/nexco-lab](https://github.com/YuqiWang-code/nexco-lab)。

## 已知问题

TSP100 专用环境及训练、验证数据已准备，模型训练所得检查点保存在服务器。其他规模和任务的数据准备状态及代码限制见 [项目现状](docs/项目现状.md)。
