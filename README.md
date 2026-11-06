# RLNN Lab

基于 Shengyu Feng、Yiming Yang 的 [RLD4CO 官方实现](https://github.com/Shengyu-Feng/RLD4CO)，研究 RLNN 神经组合优化方法。当前起点为 **MIS、RB200–300、单张 RTX 5090**，版本为 `v000_rlnn_baseline`。

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

源码、文档和管理工具已迁移。当前没有正式 RLNN 训练或评价成绩，`docs/实验结果.xlsx` 仅保留表头。Windows 缺少 PyG、torch-scatter、Accelerate、tqdm；完整 RLNN 冒烟测试尚未执行成功。服务器部署目录和新环境待准备，自动同步已关闭。

服务器 Linux（部署路径确认、源码同步完成后，从项目根目录执行）：

```bash
python3.11 -m venv "$HOME/envs/rlnn"
source "$HOME/envs/rlnn/bin/activate"
bash train_scripts/v000_rlnn_baseline/setup_env.sh
RUN_NAME=smoke_001 bash train_scripts/v000_rlnn_baseline/smoke.sh
bash train_scripts/v000_rlnn_baseline/prepare_data.sh
RUN_NAME=rb_train_seed0 bash train_scripts/v000_rlnn_baseline/train.sh --seed 0
CHECKPOINT_DIR="$PWD/outputs/v000_rlnn_baseline/rb_train_seed0/checkpoints" RUN_NAME=rb_test_seed0 bash train_scripts/v000_rlnn_baseline/eval.sh --seed 0
```

默认 `CUDA_VISIBLE_DEVICES=1`，使用前确认该设备当前可用；可显式覆盖。训练入口固定 `--method RLNN --loss erdoes --num_l 5 --num_h 128`，以单个 Python 进程运行。数据集与 checkpoint 配置见[数据集](docs/数据集.md)和[基线](docs/RLNN基线.md)。环境安装方案尚未在服务器执行。

Windows 本地整理（拉回真实运行产物后；使用提供 `@oai/artifact-tool` 的 Node 运行时）：

```powershell
python tools/server.py pull
node tools/collect_results.mjs
python tools/snapshot.py --version v000_rlnn_baseline
```

同步目前会拒绝执行，直至本地 `.vscode/sftp.json` 中的新部署目录获确认，填写 `remotePath` 并设 `deploymentConfirmed=true`。凭据不入库。工具依赖及冒烟结果的单独整理方式见 [tools/README.md](tools/README.md)。

基线论文：Shengyu Feng and Yiming Yang. *Regularized Langevin Dynamics for Combinatorial Optimization*. ICML 2025, PMLR 267:16977–16992. [正式论文页](https://proceedings.mlr.press/v267/feng25r.html)。[本地 PDF](docs/参考文献/神经组合优化/RLNN%28ICML2025%29.pdf)及[源码来源](docs/源码来源.md)保留原始归属。
