# RLNN Lab

基于 Shengyu Feng、Yiming Yang 的 [RLD4CO 官方实现](https://github.com/Shengyu-Feng/RLD4CO)，研究 RLNN 神经组合优化方法。当前起点为 **MIS、RB200–300、单张 RTX 5090**（13 图合批，每图 1000 状态），版本为 `v000_rlnn_baseline`。

[项目仓库](https://github.com/YuqiWang-code/RLNN_lab) · [基线配置与差异](docs/RLNN基线.md) · [项目现状](docs/项目现状.md) · [文献索引](docs/参考文献/文献索引.md)

| 目录 | 用途 |
| --- | --- |
| `models/` | 上游源码及后续模型迭代；保留原始布局 |
| `train_scripts/v000_rlnn_baseline/` | 环境、数据、训练、评估与冒烟入口 |
| `outputs/v000_rlnn_baseline/<run>/` | 实际日志、配置和指标；权重留在服务器 |
| `docs/` | 服务器、数据、基线说明、参考文献和实验结果工作簿 |
| `docs/temporary/` | 当前想法、代码与工作簿数值快照 |
| `tools/` | 运行记录、同步、结果整理与快照工具 |
| `tmp/` | 可清理的本地临时材料，不提交 |

源码迁移与独立服务器部署已完成，GPU 1 的训练、解码与 fp16 冒烟均已通过。RB train/val/test 1000/100/500 图已准备。张量缓存与 8 worker 的续训正在运行，目标为总第 50 epoch；独立测试尚未完成。

服务器 Linux（独立部署目录 `/home/hzeng/project/ZH/RLNN_lab`，从项目根目录执行）：

```bash
python3.10 -m venv "$HOME/envs/rlnn"
source "$HOME/envs/rlnn/bin/activate"
bash train_scripts/v000_rlnn_baseline/setup_env.sh
RUN_NAME=smoke_001 bash train_scripts/v000_rlnn_baseline/smoke.sh
bash train_scripts/v000_rlnn_baseline/prepare_data.sh
bash train_scripts/v000_rlnn_baseline/prepare_cache.sh
RUN_NAME=rb_train_seed0 bash train_scripts/v000_rlnn_baseline/train.sh --seed 0
CHECKPOINT_DIR="$PWD/outputs/v000_rlnn_baseline/rb_train_seed0/checkpoints" RUN_NAME=rb_test_seed0 bash train_scripts/v000_rlnn_baseline/eval.sh --seed 0
```

默认 `CUDA_VISIBLE_DEVICES=1`，使用前确认该设备当前可用；可显式覆盖。训练入口固定 `--method RLNN --loss erdoes --num_l 5 --num_h 128`，以单个 Python 进程运行。数据集与 checkpoint 配置见[数据集](docs/数据集.md)和[基线](docs/RLNN基线.md)。服务器环境已安装并通过 GPU 冒烟。

Windows 本地整理（拉回真实运行产物后；使用提供 `@oai/artifact-tool` 的 Node 运行时）：

```powershell
python tools/server.py pull
node tools/collect_results.mjs
python tools/snapshot.py --version v000_rlnn_baseline
```

本地 `.vscode/sftp.json` 已确认独立 RLNN 部署路径，`deploymentConfirmed=true`；自动保存上传关闭，按需同步。凭据不入库。工具依赖及冒烟结果的单独整理方式见 [tools/README.md](tools/README.md)。

基线论文：Shengyu Feng and Yiming Yang. *Regularized Langevin Dynamics for Combinatorial Optimization*. ICML 2025, PMLR 267:16977–16992. [正式论文页](https://proceedings.mlr.press/v267/feng25r.html)。[本地 PDF](docs/参考文献/神经组合优化/RLNN%28ICML2025%29.pdf)及[源码来源](docs/源码来源.md)保留原始归属。
