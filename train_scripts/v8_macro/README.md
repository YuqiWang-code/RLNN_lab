# v8 宏步（ACMD）启动器

`models/` 侧对应文件：`mis_v8_macro.py`、`v8_collect.py`、`v8_train.py`、`v8_eval.py`、`check_v8.py`、`diagnose_v8_prior.py`。方案与判读标准见 [RLNN_lab_v8_review_and_research.md](../../docs/temporary/RLNN_lab_v8_review_and_research.md)，实测结果与"停止主线"的决定见 [RLNN_v8_P0_Results.md](../../docs/temporary/RLNN_v8_P0_Results.md)。

**现状：P0 闸门已判定停止，训练与评测脚本保留但未产出结论。** 下面的顺序是完整路径，重启时按序执行即可；第 2 步是已知的止损点。

## 顺序

| 步 | 命令 | 作用 |
| --- | --- | --- |
| 0 | `bash check.sh` | 内核自检：M=1 读出等值、联合似然退化、块记录语义（CPU，无检查点） |
| 1 | `bash collect.sh` | 冻结教师切块，写 `outputs/v8_macro/data/blocks_{train,validation}_ref0.pt` |
| 2 | `bash prior.sh` | P0-B 相关性打乱 + P0-C 端点档案漏失（约 2.5 分钟 / 60 图） |
| 3 | `bash train_all.sh` | A2 / A3 / A5 三臂小训练（先读结果文档里的损失标定问题） |
| 4 | `bash eval.sh A3 budget` | 固定预算评测，两臂同图同种子，冻结 PAVE `horizon=1` |

## 定位与步数（现行方向）

宏步线停在 P0 之后，这条目录改用来跑「等预算定位 + 步数杠杆」，结论见
[RLNN_v8_Positioning_Results.md](../../docs/temporary/RLNN_v8_Positioning_Results.md)，汇总表在
`outputs/v8_macro/positioning/summary/positioning_ref0.md`。

| 步 | 命令 | 作用 |
| --- | --- | --- |
| P0 | `bash probe_step.sh` | 每步成本分解（eager / captured / forward / energy / decode），约 2 分钟 |
| P1 | `bash check_capture.sh` | 捕获闸门：同种子 20 步必须逐位一致，然后才报速度（2 图，约 1 分钟） |
| P2 | `bash kamis.sh` | KaMIS anytime 曲线，4 档 `--time_limit` × 2 种子（纯 CPU，约 42 分钟） |
| P3 | `bash capture_budget.sh` | eager vs captured 等时钟对照，0.5/1.0 秒（GPU，约 11 分钟） |
| P4 | `BUDGETS="2.5 5.0" ARMS=captured RESULT_DIR=outputs/v8_macro/positioning_wall bash capture_budget.sh` | 等钟高预算档（约 45 分钟） |
| P5 | `python models/summarize_v8_positioning.py` | 汇总三份 CSV 成一张定位表 |

`models/` 侧对应文件：`mis_v8_capture.py`、`check_capture_budget.py`、`profile_budget_step.py`、
`eval_kamis_anytime.py`、`eval_v8_capture_budget.py`、`summarize_v8_positioning.py`。

## 环境变量

`common.sh` 里统一设置，全部可覆盖：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `PYTHON` | `/home/hzeng/envs/rlnn/bin/python` | 服务器解释器（本地无 `torch_scatter`，不能在 Windows 上跑） |
| `OUT_V8` | `outputs/v8_macro` | 产物根目录 |
| `REFERENCE_SEED` | `0` | 骨干与 PAVE 的参考种子，必须成对 |
| `SEEDS` | `1017 1019 1021` | 搜索种子；`diag*`/`collect` 只用第一个 |
| `HORIZON` | `2` | 块的宏步长度；PAVE 侧固定 1 |
| `CUDA_VISIBLE_DEVICES` | `0` | 两张 5090 长期被占用，换卡前先看 `nvidia-smi` |
| `EPOCHS` | `3` | 单臂训练轮数 |
| `GRAPHS` | `100` | 定位实验用的验证图数 |
| `BUDGETS` | `0.5 1.0` | 预算档（秒），空格分隔 |
| `ARMS` | `eager captured` | 引擎臂；只跑捕获臂时写 `ARMS=captured`，此时不出配对行 |
| `REPEATS` | `2` | 每格的重复次数，两臂顺序交替 |
| `RESULT_DIR` | `$OUT_V8/positioning` | 定位产物根目录，高预算档写 `$OUT_V8/positioning_wall` |
| `SEEDS_LIST` | `1017` | 定位实验的搜索种子（与 `SEEDS` 分开，避免误用三个种子把时长乘三） |
| `KAMIS` | `$ROOT/tmp/kamis_build/redumis` | ReduMIS 二进制；缺失时 `kamis.sh` 直接报错 |

## 注意

- `HORIZON=2` 是**宏步长度**，与 PAVE 的 `horizon` 不是一回事。评测器里 PAVE 一律 `horizon=1`，改错会让结论与 v5/v7 基线不可比（v5 归档的 `fixed_time` 运行全是 1，v7 普查曾误用 2，已记录为偏差）。
- 训练必须把 `edge2adj` 的归一化邻接喂给 kernel，不是 0/1 邻接；这是 v8 首轮训练浪费掉一轮的直接原因。
- `prior.sh` 只用冻结模型，不训练、不写检查点，可以随时重跑；它是这条线的止损闸门，先跑它再谈训练。
- 定位实验一律在 `CUDA_VISIBLE_DEVICES=1` 上跑：GPU0 长期被其他项目占满，争用会让 `ms_per_step` 和 `deadline_overrun` 失真。
- `kamis.sh` 是纯 CPU 任务（`CUDA_VISIBLE_DEVICES=""`），只传 `--time_limit`，但 ReduMIS 每次调用的墙钟地板是 1.87 s/图，
  所以「0.1 秒档」在墙钟上并不存在；表里必须同时报解和实测墙钟，不能只报 `--time_limit`。
- `capture_budget.sh` 的 `BUDGETS`/`ARMS` 是**不加强引号**展开的（空格分隔列表）；加引号会被当成单个参数，报
  `argument --budgets: invalid float value: '0.5 1.0'`。
