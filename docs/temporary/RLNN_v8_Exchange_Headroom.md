# 交换头的信号检查：半径 ≤ 4 的交换在收敛解上不存在

> 2026-11-15。路线第 2 步（学习型交换头）在动手训练之前要先回答一个问题：
> **还有没有「交换就能涨」的移动可学？** 这个探针的答案是「没有」，所以第 2 步以
> 另一种方式被判死：不是学不好，是**没有可学的正样本**。

## 0. 为什么先查这个

PAVE 的交换邻域其实已经不小。`models/mis_v2_exchange.py:48 _exchanges(selected, size)`
在 `size=1` 时枚举「删 1 加 1」（增益 0）和「删 1 加 2」（增益 +1），`size=2` 时枚举
「删 2 加 2」（增益 0）和「删 2 加 3」（增益 +1）。`models/mis_v3_search.py:222
improving_actions` 会把这两种都取出来。而实测这些交换在收敛解上**一次都不发生**
（KaMIS 起点 100/100 图零交换、我们自己的 1.0 s 解也是零）。

于是唯一可能的余地是**更大的移动**：k → k+1，k ≥ 3。这些当前管线里没有任何组件枚举过。
学习型交换头的价值完全取决于「这些移动存在且能被挑出来」。

## 1. 探针

`models/probe_exchange_headroom.py`：

- 用冻结骨干 + 捕获引擎跑出 0.2 s / 1.0 s 的部署解（搜索 + PAVE），拿到真实位集；
- 对解 S，把自由点按「它在 S 里的邻居集合」（blockers）分组；
- 候选释放集 R 取：单个 blockers、两个 blockers 的并、以及这些并再并上单个已选点
  （保证 |R| ≤ k 时不会漏掉真正的 k-半径移动）；
- 对每个 R：所有 blockers ⊆ R 的自由点构成 pool，在 pool 里贪心取独立集，
  若大小 > |R| 就是一个 k → k+1 的见证。

**正对照**（`--control`，用最小度贪心解当输入，6 图）：k1 2/6 图、k2 3/6 图、k3 3/6 图，
单图最多 k1:1 / k2:8 / k3:30 个见证 —— 探针确实能看见移动。

## 2. 结果（1.0 s 部署解，20 图冒烟）

```
summary
1.00s graphs 20 pave 19.7000 ones/exact 27.40 | k2: 0/20 | k3: 0/20 | k4: 0/20
```

`ones/exact 27.40` 是「只被一个已选点挡住」的自由点数——它有 27 个，但**没有一对**
能同时塞进去（它们两两相邻），所以 PAVE 的 (1,2) 邻域也是空的。

也就是说：部署解在 **1-flip、(1,1)、(1,2)、(2,3)、k≤4 的全部 k→k+1** 上都动了不了。

完整扫描（60 图 × 0.2/1.0 s，`outputs/v8_macro/exchange_headroom/headroom/headroom_ref0.csv`）：

| 时钟 | PAVE 均值 | ones/exact | k1 有见证 | k2 | k3 | k4 |
| 0.20 s | 19.7500 | 29.57 | 0/60 | 0/60 | 0/60 | **1/60** |
| 1.00 s | 19.8833 | 28.00 | 0/60 | 0/60 | 0/60 | 0/60 |

60 图里唯一的例外是 0.2 s 档的 1 张图（k=4 有一个 +1 的移动），1.0 s 档是
**干净的全零**。这条结论不是 20 图冒烟的抽样噪声。

## 3. 判读

- **学习型交换头没有正样本可学**：它要做的事是「在许多增益 +1 的交换里挑最好的」，
  而这些交换根本不存在。训练它只会学到「不要动」，或者在噪声上过拟合。
- 同分布上 KaMIS 1.0 s 能到 20.10、我们 19.99，差 0.11 是**真的**，但差距不在「挑交换」
  这一步，而在**怎么走到另一个盆地**：要么允许零增益的游走（退火/接受上坡），要么做
  缩减（ReduMIS 的 reduction + 重启），要么一次性移动 ≥5 个点。
- 这条结论和之前三条负结果（末端 PAVE +0.004、König 重组 +0.024、循环内交换 ≤0）拼起来，
  完整解释了 19.98 这个天花板：**它不是「搜索没跑够」，而是「当前移动集闭合」**。

## 4. 下一步

按 README 路线，第 2 步作废，直接进第 3 步：**学习式退火**——把「接受上坡」学出来，
让搜索能在零增益/负增益的状态走一段再出来。这条的依据是论文自己：纯 SA 的 RLSA
（19.97 @ 0.070 s）比他们自己的神经网络版 RLNN（19.52 @ 0.197 s）强 0.45 且快 2.8 倍，
说明「贪心接受」本身就是短板；我们要做的不是把贪心调快，而是学一个**比固定 τ 更好的
接受准则**（价值头 `V(s)` 预测这条轨迹最终的档案大小，接受 `ΔV > −τ(t)·log u`）。

## 5. 复现

```bash
python models/probe_exchange_headroom.py --split test \
  --test_path models/data/MIS_tensors/RB_small_author_test --test_samples 60 \
  --budgets 0.2 1.0 --radii 1 2 3 4 --reference_seed 0 --seeds 1017 \
  --save_dir "$PWD/outputs/v2/none_seed0/checkpoints" \
  --result_dir outputs/v8_macro/exchange_headroom

# 正对照：把搜索换成最小度贪心解
python models/probe_exchange_headroom.py --split test \
  --test_path models/data/MIS_tensors/RB_small_author_test --test_samples 6 \
  --budgets 1.0 --radii 1 2 3 --control --reference_seed 0 --seeds 1017 \
  --save_dir "$PWD/outputs/v2/none_seed0/checkpoints" \
  --result_dir outputs/v8_macro/exchange_control
```

## 6. 产物

- `outputs/v8_macro/exchange_headroom/headroom/headroom_ref0.csv`（60 图 × 2 时钟，逐图 `ones`/`moves_k`）
- `outputs/v8_macro/queue/exchange_headroom.log`
- `outputs/v8_macro/exchange_control/headroom/headroom_ref0.csv`（正对照 6 图）
