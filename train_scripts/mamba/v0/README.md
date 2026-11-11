# Mamba v0 · PC-OSM-RLNN

Mamba 研究线的第一版，对应 [v7 方案](../../../docs/已归档的想法和文件/mamba/RLNN_v7_OneShot_Mamba_Research_and_Implementation_Plan.md)。模型使用冻结的 B 和 PAVE H1。一次性编码在求解开始时完成，节点动作偏置在 20 条粒子和所有采样步间共享。训练每次更新重新计算带梯度的编码。[P0 结果](../../../docs/已归档的想法和文件/mamba/RLNN_Mamba_v0_P0_Results.md)

```bash
CUDA_VISIBLE_DEVICES=0 bash train_scripts/mamba/v0/setup_env.sh
export RLNN_SOURCE_SHA="$(git rev-parse HEAD)"
CUDA_VISIBLE_DEVICES=0 bash train_scripts/mamba/v0/p0.sh correctness
CUDA_VISIBLE_DEVICES=0 bash train_scripts/mamba/v0/p0.sh timing
CUDA_VISIBLE_DEVICES=1 bash train_scripts/mamba/v0/p0.sh information
CUDA_VISIBLE_DEVICES=1 bash train_scripts/mamba/v0/diagnose.sh
python train_scripts/mamba/v0/decide.py --stage p0 --output outputs/v7_context/analysis/p0.json
```

源代码通过 SFTP 同步且服务器目录没有 Git 元数据时，`RLNN_SOURCE_SHA` 使用同步版本的提交号。RTX 5090 的安装脚本为 SM120 构建官方 Mamba-1 CUDA 扩展，复用 RLNN 基础环境并隔离新增依赖。

通过 P0 后，按种子分别训练四个核心组。环境变量 `REF_SEED=0/1/2` 分别对应 42/123/999，`WIDTH` 默认 32。每个进程用一张卡。正式测速在各卡独占运行。

```bash
CUDA_VISIBLE_DEVICES=1 REF_SEED=0 bash train_scripts/mamba/v0/train.sh v7_mamba
CUDA_VISIBLE_DEVICES=0 REF_SEED=0 bash train_scripts/mamba/v0/eval.sh v7_mamba --cuda-graph
CUDA_VISIBLE_DEVICES=0 REF_SEED=0 BUDGET=0.5 bash train_scripts/mamba/v0/eval_budget.sh v7_mamba --cuda-graph
```

核心对照为 `baseline`、`v7_mlp`、`v7_message`、`v7_shuffle_mamba`。线性、度排序、单向和 logit 门控可由 `--context_mode` 切换，`--v7_order_views` 设为 1、2 或 4。固定工作量和两档时间预算的逐图结果均写入 `outputs/v7_context`。

冻结 B 和 PAVE 的静态先验小试使用 `pilot_static.sh`。首轮完成后用 `pilot_static.sh repeat-budgets` 复测两档预算，再运行 `python train_scripts/mamba/v0/analyze_static_pilot.py` 汇总逐图差值。[小试与 v1 方案](../../../docs/已归档的想法和文件/mamba/RLNN_Static_Prior_v1_Analysis_and_Plan.md)
