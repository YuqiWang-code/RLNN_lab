# RLNN 基线

版本 `v000_rlnn_baseline`，MIS RB200–300，单张 RTX 5090，单进程。依据 Feng、Yang 的 ICML 2025 *Regularized Langevin Dynamics for Combinatorial Optimization*。

## 配置与预算

| 项目 | 论文 / 作者原始配置 | 本项目单卡入口 | 冒烟 |
| --- | --- | --- | --- |
| 方法 / 损失 | RLNN / erdoes | RLNN / erdoes | 相同 |
| 网络 | 5 层 GCN，隐藏 128 | 相同 | 相同 |
| epochs / Adam lr | 50 / 0.0001 | 相同 | 1 / 0.0001 |
| 状态 batch | 论文每 GPU 32；脚本 64 | 32 | 2 |
| GPU 数 | 作者脚本 8 | 1 | 1（无 GPU 时可用 CPU） |
| K / T | 20 / 100 | 20 / 100 | 2 / 2 |
| K′ / T′ | 论文 10 / 50；脚本 num_kp=10 | CLI 10 / 50，实际使用 K=20 | 实际 2 / 2 |
| d / β / λ | 5 / 1.02 / 0.5 | 相同 | 相同 |
| train / val | 1000 / 100 | 相同 | 2 / 1 |
| test | 500 图一组，论文平均两组 | 本地生成 500，未与作者集对齐 | 1 |
| 训练 / 推理精度 | float32 / float16 | no / fp16 | no / no |

上游 `generate_data` 使用 `num_k` 初始化粒子，`num_kp` 仅保存未引用；每图得到 `num_k*num_tp=1000` 状态，单卡 batch 32 时每图 32 次更新、完整 1000 图每 epoch 32000 次更新。论文 K′=10 对应 500 状态，语义不相同。本项目保留 ZIP 行为，不称为严格论文复现。

DataLoader 的 graph batch 固定为 1；CLI `batch_size` 是一张图内轨迹状态的批大小。假设完整整除切分、每卡 125 图，作者八卡脚本每图 ceil(1000/64)=16 次同步更新，每 epoch 约 2000 次同步更新；本项目单卡约 32000 次更新。单卡调整硬件和状态批大小，没有模拟原多卡有效 batch。正式开跑前先测单卡成本，再决定后续对齐实验。

## 工程适配

`main.py` 提取 `build_parser()`，参数与算法不变。`tools/run_rlnn.py` 复用实际解析器，显式覆盖项目默认值，进入 `models/` 执行 `main.py`。记录 stdout/stderr、完整命令、配置、可见 GPU、框架与数据路径数量，成功后提取真实指标。运行名目录必须不存在，避免覆盖。`--checkpoint` 指向含 `best.pt` 的目录。

作者脚本使用 `--train_sample` 等 argparse 缩写，并引用未定义 `LOGDIR`；项目脚本使用完整参数和固定输出目录。真实参数是 `--lambd`。原脚本保留作参考，不直接用作项目入口。

训练脚本可用 `TRAIN_PATH`、`VALID_PATH`、`TEST_PATH`、`RUN_NAME`、`PYTHON`、`CUDA_VISIBLE_DEVICES` 配置，额外参数传给上游解析器，如 `--epochs 1 --train_samples 2`。相对数据路径按项目根目录解释；checkpoint 推荐绝对路径。`reinforce` 是另一种损失，上游说明未用于论文报告；RLSA 是另一求解器。

## 指标

MIS 原始能量为 `β∑(u,v)∈E x_u x_v − ∑u x_u`，每条有向边权重 1/2。训练 `erdoes` 损失为软翻转后能量均值，加 `λ(∑p−d)²` 正则；日志 `loss` 是跨批次指数平滑值（0.95/0.05），不是独立集大小。

默认启用解码：每个粒子保存历史最低能量二值状态，再按该状态优先级贪心添加节点、屏蔽邻居，得到可行独立集。上游返回负独立集大小，并取粒子最小负值，再求数据集均值。`val/raw_obj` 保留 `valid_obj`；`test/raw_obj` 保留 `Avg. obj`；派生 `independent_set_size=-raw_obj`，越大越好。禁用解码得到惩罚能量，不能按相同规则解读，项目启动器要求解码开启。

`val/time`、`test/time` 是上游每图 `evaluate_single` 的平均墙钟秒数，包含采样与解码，不包含完整训练，且未显式 CUDA 同步。`run/wall_time` 是外围进程总耗时，包含启动、训练和验证（冒烟还包含测试），单独记录。评估原始文件仅保留 6 位小数。

当前没有可靠参考值，不产生 gap。工作簿 notes 记录 kind、seed、完整配置、设备与原日志目录；run.json 保存来源。正式单卡与 smoke 分别进入不同工作簿。作者 `logs_train/` 不参与收集。

## 验证状态与已知问题

尚未完成 GPU 环境、RB 数据准备、端到端冒烟、正式训练或评价。论文附录 A.3 描述按能量梯度迭代翻转后处理，与 ZIP 中按候选状态排序构造独立集不同；保留 ZIP 解码。作者多卡耗时不作为本项目单卡实测值。

上游训练日志 `Iteration` 未随更新递增，因此不导入为训练步数。`MetaSolver.prepare` 将返回优化器写入拼写为 `optimzier` 的属性，训练仍使用原 `optimizer`；单进程保留原行为，多卡训练需另行核对，本版本不启用。

上游 main.py 的 RLPGSA 分支引用未定义的类，不能作为可运行求解器；当前项目入口限定 RLNN。
