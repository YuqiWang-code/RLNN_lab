# v5 DAFR-Lite

DAFR-Lite 在 RLNN 的最终翻转 logits 上添加 112 参数的标量适配器。先将当前二值状态、邻居计数、度数、占用比例和精确局部能量差编码为两路标量并门控，再用同一次稀疏传播得到惰性低通与互补高通。MLP 对照只计算邻居计数，MSG 对照使用相同投影调用并保留局部项。原 RLNN 参数为 165,505，三种适配器共 165,617 参数。

从仓库根目录执行，`PYTHON` 指向服务器的 RLNN 环境。

| 入口 | 用途 |
|---|---|
| `smoke.sh` | CPU/CUDA 自检及 8 图训练冒烟 |
| `screen.sh` | 两卡各 6 个单卡任务，四臂 × 三种子，原冻结教师和原 erdoes |
| `train.sh ARM` | 单臂训练，`SEED` 指定种子 |
| `eval.sh ARM` | 默认验证集 100 图，FP32 固定步数，`PRECISION=fp16` 可改精度 |
| `eval_budget.sh ARM` | `BUDGET=0.5/1.0`，按种子使用冻结 PAVE 或 `POST=rule23` |
| `diagnose.sh` | 共享随机数回放，`--initial` 选择零启动，`--utility` 选择训练图 R8 收益探针 |
| `profile.sh` | 空闲单卡上测 20 图 × 5 次，分别保存真实总时间和组件微基准 |
| `validate.sh` | 等待训练和预诊断完成，两卡各一条评测序列 |
| `followup.sh` | GPU 1 运行诊断和触发的冲突掩码训练，GPU 0 在预算评测后做剖析和固定步数验收 |

解析对照使用 `eval.sh baseline --calibrate-analytic`。每个种子在首 20 张训练图上比较 0、0.0625、0.125、0.25 四个系数，Size 相同取更小系数。生成的 `calibration_seedN/run.json` 供 `eval.sh analytic` 使用。

短筛用同种子 v2 骨干热启动，训练 1000 图、验证 100 图，每轮 2000 次更新。适配器第 1 轮冻结骨干，第 2–3 轮以 2e-5 解冻。原骨干微调对照第 1 轮训练骨干。各臂共享冻结教师状态生成协议和同种子随机流。

`python train_scripts/legacy/v5_frequency/decide.py` 生成 `outputs/v5_frequency/analysis/` 的汇总、图级配对与准入判断。FP32 固定 K20/T100 的无后处理 Size 为主质量口径，FP16 和 0.5/1.0 秒分别统计。验证集用于权重选择。大体积检查点和原始诊断明细保留在服务器。
