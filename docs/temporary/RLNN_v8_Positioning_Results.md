# RLNN v8 定位实验：等预算对照与步数杠杆

## 0. 这份文档回答什么

前面九条方法线（PAVE、频域、mamba、结构补偿 v1/v2、static prior、v7 context/TCCR 等）的可复现增益全部落在零附近，
所以继续加模块已经没有杠杆。本阶段改为先回答两个**定位**问题，再决定要不要继续：

1. 与 SOTA（KaMIS/ReduMIS）在**同一个时钟**下，我们处在什么位置？此前唯一的对照是 KaMIS 固定 4.11 s/图 = 20.06，
   而我们是 19.51；没有等预算曲线，这个数字既不能证明我们快，也不能证明我们差。
2. 论文真正的技术杠杆在哪？`models/profile_budget_step.py` 给出线索：eager 每步 2.16 ms，捕获成 CUDA 图后 0.26–0.33 ms。
   冻结策略本身是我们自己的，**步数是唯一「每毫秒价值已知为正」的方向**。

## 1. 协议

- 数据：`models/data/MIS_tensors/RB_small_val` 前 100 图（`--split validation --test_samples 100`）。
- 搜索种子 1017，参考种子 ref0（冻结骨干 `outputs/v2/none_seed0/checkpoints/best.pt`，冻结 PAVE v3 H1）。
- 一律 **PAVE horizon=1**（这是 v5 归档 `run.json` 的口径；v7 误用 horizon=2，见 `docs/temporary/RLNN_v7_TCCR_Results.md`）。
- 预算协议沿用 `models/eval_v7_budget.py`：`post_reserve = 0.05 s` 留给 PAVE，搜索在 `deadline - 0.05` 前停止；
  每 10 步做一次 A.3 解码，取历史最优；两次重复，两臂顺序交替。
- 两个引擎**只差在执行方式**，轨迹必须逐位一致：`models/check_capture_budget.py` 在同一图上跑 20 步，
  断言 state / archive states / archive energy / 解码掩码全部相等，通过后才允许报速度。
  - 实测（validation 前 2 图）：`identical over 20 steps; eager 1.706 ms/step, captured 0.262 ms/step (6.51x), capture 46.4 ms`。
  - 产物 `outputs/v8_macro/check_capture/check/check_ref0.json`。
- 捕获成本（每图一次，约 46–126 ms）**计入预算**，不是免费加速。

## 2. 引擎消融：等时钟、同轨迹

100 图 × 2 时钟 × 2 引擎 × 2 重复，图级配对自助 5000 次。

| 时钟 | eager 解 | captured 解 | 配对差 (captured−eager) | 胜/平/负 | 步数比 | eager 步数 | captured 步数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 s | 19.6900 | 19.8850 | +0.1950 [+0.1250,+0.2700] | 21/79/0 | 6.75× | 121 | 795 |
| 1.0 s | 19.7750 | 20.0000 | +0.2250 [+0.1400,+0.3200] | 22/78/0 | 7.88× | 301 | 2348 |

单步成本（同一次运行实测）：eager 3.66 / 3.14 ms，captured 0.585 / 0.413 ms；捕获前导（含一次解码）
126.5 / 125.2 ms，即 0.5 s 档里有 25% 的时钟被一次性捕获吃掉，仍然换来 6.75 倍步数。

判读：

- 两个时钟都是**零负**，且 79/100、78/100 是平局——平局来自小图已经收敛，不是没效果。
- 量级对照：0.5 s 捕获后的 19.885 已经超过 v7 在 1.0 s 档的 19.823，也超过固定工作量 100 步＋PAVE 的 19.6244。
  换句话说，**同样的墙钟，先前要花两倍时间才能拿到的解，现在 0.5 s 就能给出**。
- 这是本项目在「质量」上第一个**可复现、零负、CI 不跨零**的正结果，且在 0.5 s 与 1.0 s 两个时钟同时成立。
- 机制清楚：冻结策略每步只花 0.26 ms 算术，其余都是 kernel 发射延迟；捕获把 10 步压成一次发射。
- 附带事实：captured 臂 `search_size == final_size`（PAVE 边际 0），说明步数足够后 PAVE 的后处理增益被搜索本身吸收；
  这与 v7 在 1.0 s 档观测到的 `pave_gain 0.022` 同向。

## 3. KaMIS（ReduMIS）anytime 曲线

`tmp/kamis_build/redumis`（sha256 `edf3c458…`），100 图 × 4 个 `--time_limit` × 2 个 KaMIS 种子 = 800 次调用。

| KaMIS time_limit | 平均解 | 中位解 | 实测墙钟 s/图 | 最慢图 s |
| --- | --- | --- | --- | --- |
| 0.1 s | 20.0450 | 20.0 | 1.87 | 4.82 |
| 0.5 s | 20.0600 | 20.0 | 2.67 | 6.05 |
| 1.0 s | 20.0650 | 20.0 | 3.12 | 7.67 |
| 4.0 s | 20.0650 | 20.0 | 4.93 | 8.04 |

判读：

- **时间上限几乎不改变解质量**（20.045 → 20.065，40 倍预算只值 +0.02）。RB_small_val 上 ReduMIS 在很短时间内就饱和。
- 但它的**墙钟地板是 1.87 s/图**（进程启动、图写盘、缩减预处理），即使传 `--time_limit=0.1` 也压不下去。
  所以「0.5 秒谁强」这个问题，KaMIS 这一侧根本答不了：它在 0.5 s 内交不出任何结果。
- 我们此前的 4.11 s/图 = 20.06 与这里 4.93 s = 20.065 一致，说明参考实现和口径没漂。

## 4. 等钟对照

见 `outputs/v8_macro/positioning/summary/positioning_ref0.md`（由 `models/summarize_v8_positioning.py` 生成）。
表格按「实测墙钟」对齐两种方法：KaMIS 每个 time_limit 的实际墙钟，对上我们最接近的预算档，按图配对做自助区间。

| 墙钟 s/图 | KaMIS 解 | 我们（captured）解 | 配对差（我们−KaMIS） | 胜/平/负 |
| --- | --- | --- | --- | --- |
| 1.87（KaMIS 0.1 s） vs 2.49（我们 2.5 s） | 20.0450 | 20.0400 | −0.0050 [−0.0450,+0.0350] | 2/94/4 |
| 2.67（KaMIS 0.5 s） vs 2.49（我们 2.5 s） | 20.0600 | 20.0400 | −0.0200 [−0.0600,+0.0200] | 2/93/5 |
| 3.12（KaMIS 1.0 s） vs 2.49（我们 2.5 s） | 20.0650 | 20.0400 | −0.0250 [−0.0700,+0.0150] | 2/93/5 |
| 4.93（KaMIS 4.0 s） vs 4.98（我们 5.0 s） | 20.0650 | 20.0500 | −0.0150 [−0.0500,+0.0200] | 1/96/3 |

完整曲线（同一张表，按真实墙钟排序）：

| 方法 | 设定 | 步数 | 解 | 实测墙钟 s/图 |
| --- | --- | --- | --- | --- |
| RLNN captured | 0.5 s | 795 | 19.8850 | 0.49 |
| RLNN eager | 0.5 s | 121 | 19.6900 | 0.48 |
| RLNN captured | 1.0 s | 2348 | 20.0000 | 0.99 |
| RLNN eager | 1.0 s | 301 | 19.7750 | 0.98 |
| KaMIS | 0.1 s limit | — | 20.0450 | 1.87 |
| RLNN captured | 2.5 s | 7285 | 20.0400 | 2.49 |
| KaMIS | 0.5 s limit | — | 20.0600 | 2.67 |
| KaMIS | 1.0 s limit | — | 20.0650 | 3.12 |
| KaMIS | 4.0 s limit | — | 20.0650 | 4.93 |
| RLNN captured | 5.0 s | 15661 | 20.0500 | 4.98 |

判读：

- 四格配对差**全部 CI 跨零**，93–96/100 图完全相同 → 在 KaMIS 能跑的时钟上，两边是**同一水平**，
  均值差 −0.005 ~ −0.025 而不是之前那种 −0.45。
- 0.5 s 到 1.8 s 之间是**只有我们能跑的区间**：KaMIS 的墙钟地板 1.87 s/图，这个区间里它交不出结果，
  我们给出 19.885（0.5 s）到 20.000（1.0 s）。
- KaMIS 的时间上限在 RB_small_val 上几乎无效（20.045 → 20.065），所以「KaMIS 慢一点但更准」也不成立：
  它在能跑的最快设定（1.87 s，20.045）就已经接近自己的天花板（20.065）。
- 我们的曲线仍在随步数上涨（7285 → 15661 步，20.040 → 20.050，趋饱和），而 KaMIS 早已平；
  这意味着再过一档预算（比如 10 s）我们可能反超，但边际很小，不值得为此加实验。
- 注意 `solve_time_s` 是我们的完整求解墙钟（含捕获与 PAVE），KaMIS 的墙钟含进程启动与图写盘——
  这是两者的真实开销，不做调整。

## 5. 结论

1. 定位问题有了答案：**在 0.5 s 与 1.0 s 这两个时钟上，我们是唯一能交结果的方法**（KaMIS 地板 1.87 s/图），
   解为 19.885 / 20.000；在 KaMIS 能跑的时钟（≥1.9 s）上，四格配对差全部跨零、93–96/100 图相同，
   两边落在同一水平（均值差 −0.005 ~ −0.025）。
   此前"19.51 vs 20.06"的对比是**不同时钟**的对比，不构成落后证据；同样也不能拿它当领先证据。
2. 技术杠杆问题也有了答案：**步数**。同样的时钟、同样的冻结策略、同样的解码与 PAVE，只把执行方式换成捕获图块，
   0.5 s 多 +0.195、1.0 s 多 +0.225，21/79/0 与 22/78/0，步数 6.75× / 7.88×。
   这是本项目第一个零负、CI 不跨零的正结果；代价是每图一次 46–126 ms 的捕获前导（已计入时钟）。
3. 因此后续顺序调整：**先做速度工程（捕获 + 解码开销），再谈第 10 条方法线**。
   解码现在占 search 时间的 22.4%（p90 48.6%，中位 6.92 ms/次），是捕获之后下一个值得动的地方；
   按本轮曲线，再快一倍步数大约还能值 +0.01 ~ +0.02（已接近饱和），所以解码优化的目标应当改成
   **让 0.5 s 档吃下现在 1.0 s 档的步数**，而不是追求无限步数。
4. 写作口径：不要把「19.885 @0.5 s vs KaMIS 20.06 @2.7 s」写成胜；正确写法是
   「在 0.5–1.8 s 区间只有本方法能在预算内给出解，在 ≥1.9 s 区间两者持平」，并把捕获前后的
   19.690 → 19.885 作为「同样的墙钟，多 6.7 倍步数」的证据。


## 6. 复现

```bash
bash train_scripts/v8_macro/check_capture.sh                     # 闸门：同轨迹 + 6.5x 速度
bash train_scripts/v8_macro/kamis.sh                             # KaMIS anytime（约 42 min，纯 CPU）
bash train_scripts/v8_macro/capture_budget.sh                    # eager vs captured，0.5/1.0 s
BUDGETS="2.5 5.0" ARMS=captured \
  RESULT_DIR=outputs/v8_macro/positioning_wall \
  bash train_scripts/v8_macro/capture_budget.sh                  # 等钟高预算档
python models/summarize_v8_positioning.py                        # 汇总表
```

产物：

| 路径 | 内容 |
| --- | --- |
| `outputs/v8_macro/check_capture/check/check_ref0.json` | 闸门（逐位一致 + 速度） |
| `outputs/v8_macro/profile_step/profile/profile_ref0.json` | 每步成本分解（eager / captured / forward / energy / decode） |
| `outputs/v8_macro/positioning/kamis/kamis_ref0.csv` | 800 行 KaMIS 结果 |
| `outputs/v8_macro/positioning/capture_budget/capture_budget_ref0.csv` | 800 行 eager/captured |
| `outputs/v8_macro/positioning_wall/capture_budget/capture_budget_ref0.csv` | 400 行 captured 高预算 |
| `outputs/v8_macro/positioning/summary/positioning_ref0.md` | 汇总表（本文 §4 的来源） |
