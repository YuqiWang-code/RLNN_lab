# v8 重训骨干（抬上限）：协议与已得结果

日期：2026-11-14。前置判断见 `RLNN_v8_Amortised_Capture.md`：摊薄之后 0.2 s 已经能压过
RLNN（19.848 vs 19.52），但对 RLSA 19.97@0.070 s 与 KaMIS 20.106@1.0 s 仍输。用户选择
「A：重训骨干，目标是 100–400 步内到 20.0+」。

## 0. 要回答的问题

冻结骨干的「质量–步数」曲线是什么样的，重训能不能把它整体抬起 0.1–0.3？

## 1. 冻结骨干的步数曲线（基线）

`models/eval_v8_steps.py`（摊薄引擎，按步数而不是按钟计时；`--decode_every 3`，120 张
作者测试图，seed 1017，无 PAVE）：

| 步数 | 100 | 300 | 1000 | 3000 |
| --- | --- | --- | --- | --- |
| 解 | 19.3917 | 19.6917 | 19.8083 | 19.8667 |
| 秒/图 | 0.03 | 0.10 | 0.31 | 0.92 |

曲线在 3000 步仍未完全平（每 1000 步还有 +0.03），**但 300 步与 3000 步差 0.175**——
这就是「100–400 步到 20.0」需要补的量级。

## 2. 重训的两个 pilot（都失败）

`train_scripts/v8_macro/retrain.sh`，从 v2/none_seed0 热启动，`--finetune_lr 2e-5`，1000 张
训练图，6 epochs。唯一变量是训练 rollout 长度 `--num_tp`（原设定 50；训练状态全部来自策略
自己前 50 步的轨迹，而部署要跑几千步）：

| 检查点 | 100 步 | 300 步 | 1000 步 | 3000 步 |
| --- | --- | --- | --- | --- |
| 冻结 v2/none_seed0 | 19.3917 | 19.6917 | 19.8083 | 19.8667 |
| 微调 `ft_tp50` | 19.4083 | 19.6500 | 19.8000 | 19.8750 |
| 微调 `ft_tp300` | 19.3583 | 19.6333 | 19.7500 | 19.8417 |

验证集（100 图、100 步）上的 `valid_obj`：基线 19.52/19.53/19.50，`ft_tp50` 19.53→19.56，
`ft_tp300` 19.47→19.40。**更长的 rollout 反而略差。**

原因在目标函数：`--loss erdoes`（`models/solvers.py:29 erdoes_graph_batch_loss`）只做
**一步前瞻**——把当前状态的翻转分布软化成 `flipped = (1-p)x + p(1-x)`，最小化它的能量，再
加 `λ·(E[#flips]-num_d)²` 的正则。它奖励「这一步翻哪些点最不坏」，完全不奖励「逃出局部最
优」。部署时的瓶颈恰恰是后者。`--loss reinforce` 也是一步 REINFORCE（奖励 = 立即能量下
降），同样短视。

## 3. 贪心初始化的负结果

同一步数评测，`--init greedy`（每个粒子从最小度贪心解出发，30/100/300 步）：
**18.7583 / 19.1750 / 19.4417**，比随机初始化分别低 0.45–0.56。策略是在随机初始化的状态
分布上训练的，换初始点反而是分布外。

## 4. 进行中

`scratch_tp300`：从零训练、`--num_tp 300`、30 epochs、1000 张图（约 2.4 小时），是「重训」
字面意义上的最后一条臂。

## 5. 判读（截至现在）

- 用现成目标函数重训，抬不动曲线；这不是超参问题，是目标函数只做一步前瞻。
- 真要「100–400 步到 20.0」，需要**改变训练信号**（序列级回报 / 逃逸奖励），或者
  **改变搜索本身**（接受温度、重启/踢动、交换邻域进循环）。
- 一个待验证的具体猜想：把 PAVE 的交换**插进搜索循环**（每 N 块解一次 → 交换 → 把结果
  写回粒子 → 继续），既可能抬上限，也让「捕获 + 平台」从替代关系变成互补关系。
  → 已测，**负结果**，见 §6。

## 6. 搜索侧的两个逃逸尝试（都失败）

§5 的猜想是「把交换或踢动插进循环」能抬上限。`models/eval_v8_interleave.py` 用摊薄引擎在
同一个时钟里对照四种调度（都走同一批图、同一个种子）：

- `search`：纯捕获搜索到底（baseline）。
- `inject`：每 `--cycle_steps` 步停一次，解码档案 → 跑 PAVE（占 `--pave_share × budget`
  的时钟）→ 若变好就把结果注入**第 0 号粒子**（其它粒子保留自己的盆地）→ 继续。
- `refill`：同上，但**所有**粒子都从交换结果重启。
- `kick`：不做交换，改成「从最优解里随机删掉 `--kick_fraction` 的点」再让搜索补洞
  （ARW 式扰动），几乎不花时钟。

30 张作者测试图、seed 1017 的探针（`outputs/v8_macro/interleave_probe_noreserve`、
`outputs/v8_macro/interleave_kick_probe`）：

| 时钟 | search | inject | refill | kick |
| --- | --- | --- | --- | --- |
| 0.05 s | 19.3667 | 19.4000 | 19.3667 | 19.3667 |
| 0.10 s | 19.5667 | 19.5667 | 19.5333 | 19.5333 |
| 0.20 s | 19.6667 | 19.5667 | 19.6000 | 19.6333 |
| 1.00 s | 19.7000 | — | — | 19.6667 |

配对差（对 `search`）：`inject` +0.033 / 0.000 / −0.100；`refill` 0.000 / −0.033 / −0.067；
`kick` +0.067 / −0.033 / 0.000 / −0.033。**没有一个格子显著为正。**

机制很清楚：交换轮里 PAVE 的 `exchanges` 几乎全是 0，`exchange_gain` 累计只有 0.03–0.17，
而在 0.2 s 档 PAVE 吃掉了 52 ms（26 % 的时钟）却没换来解 —— 花掉的步数比赚到的点多。
`kick` 连时钟都不花，58 次扰动后仍然是 0。

三条独立证据指向同一件事：**20 个粒子的档案并集已经是一个「1-flip + (2,1)-swap 双邻域
都空」的局部最优**，所以
- 末端 PAVE 在 1 s 只有 +0.004；
- König 精确重组只有 +0.024；
- 把交换塞进循环是负的；
- 随机踢动是零。

这个局部最优不是「差一点没收敛」，它就是当前移动集的上限。要越过它需要换移动集
（(2,3)-swap、缩减规则，即 ReduMIS 那一路），而不是换调度。

## 7. 复现

```bash
# 冻结骨干的步数曲线
python models/eval_v8_steps.py --split test \
  --test_path models/data/MIS_tensors/RB_small_author_test --test_samples 120 \
  --steps 100 300 1000 3000 --result_dir outputs/v8_macro/steps_baseline
# 微调
GRAPHS=1000 EPOCHS=6 NUM_TP=300 RUN_NAME=ft_tp300 bash train_scripts/v8_macro/retrain.sh
# 从零训练
WARM_START=0 EPOCHS=30 NUM_TP=300 RUN_NAME=scratch_tp300 bash train_scripts/v8_macro/retrain.sh
# 交换 / 踢动调度（1000 张作者图，单进程）
GRAPHS=1000 BUDGETS="0.05 0.1 0.2 1.0" CYCLE_STEPS=60 PAVE_SHARE=0.05 \
  ARMS="search inject" bash train_scripts/v8_macro/interleave.sh
GRAPHS=1000 BUDGETS="0.05 0.1 0.2 1.0" CYCLE_STEPS=50 ARMS="search kick" \
  RESULT_DIR=outputs/v8_macro/interleave_kick bash train_scripts/v8_macro/interleave.sh
```
