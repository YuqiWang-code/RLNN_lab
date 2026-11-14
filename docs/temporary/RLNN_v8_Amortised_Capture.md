# v8 · 捕获摊薄：一次捕获服务整个数据集

## 0. 问题

`outputs/v8_macro/author_short` 暴露的缺陷：每张图都要重新捕获一次搜索块，而捕获本身要
93.5 ms。于是 0.197 s（RLNN 的每图时钟）里有 47 % 花在准备上，0.1 s 与 0.05 s 两档
**一步都走不了**：

| 每图时钟 | 解 | 步数 | prelude |
| --- | --- | --- | --- |
| 0.05 s | 17.6720 | 0 | 93.4 ms |
| 0.10 s | 17.9947 | 0 | 93.9 ms |
| 0.20 s | 19.3970 | 49.8 | 93.5 ms |

捕获的**形状**只取决于 `(n_max, num_k)`，与具体哪张图无关；变的是邻接与 bias。所以把缓冲
按数据集的 `n_max × capacity` 一次分配、一次捕获，之后每张图只改写边缓冲，prelude 就该掉到
毫秒级。

## 1. 实现

`models/mis_v8_pad.py`：`PaddedSearch`。

- 固定 CSR 布局：`crow = arange(0, rows*cap+1, cap)`（恒定），`col`/`raw`/`norm` 是缓冲，
  `adjacency` 与 `normalized` 两个 `torch.sparse_csr_tensor` **共享**同一份 `col`。
- `load(graph, edges, seed)`：把该图写进缓冲（`dense` → 每行邻居按列升序 + 自环 + 尾部空槽）、
  重算归一化权重（与 `edge2adj` 同一表达式 `inv[row] * inv[col]`，`inv = (deg+1)^-0.5`）、
  重置档案与随机数种子。发生在时钟内 —— 这是每图真实要做的工作。
- `_body()`：与 eager 逐行一致（`flip_logits` → `rand` → `where` → energy → 档案）。
  随机数用**一个持久的** `torch.Generator`，捕获时 `register_generator_state`，每图
  `manual_seed(seed)` 而不是换对象 —— `torch.rand` 对固定 `num_k` 的形状前缀相等，所以
  padded `(n_max, k)` 与真实 `(n, k)` 抽到的是同一串数。
- 捕获发生在构造时（**一次**，不在任何一张图的时钟里）；`run()` 与 `CapturedSearch.run`
  同协议。

### 1.1 闸门（`models/check_padded_capture.py`，`train_scripts/v8_macro/check_padded.sh`）

4 张图 × 60 步：

```
1/4 0: capture identical over 60 steps (vs unpadded: 82 node-slots drifted, size 21 -> 21), load 3.8 ms, replay 0.282 ms/step, decode 3.81 ms
2/4 1: capture identical over 60 steps (vs unpadded: 66 node-slots drifted, size 19 -> 19), load 4.6 ms, replay 0.285 ms/step, decode 3.77 ms
3/4 2: capture identical over 60 steps (vs unpadded: 100 node-slots drifted, size 22 -> 22), load 8.9 ms, replay 0.280 ms/step, decode 3.54 ms
4/4 3: capture identical over 60 steps (vs unpadded: 178 node-slots drifted, size 22 -> 22), load 3.7 ms, replay 0.282 ms/step, decode 3.71 ms
padded engine gate passed: bit-identical trajectory, replay and decode
```

两条断言分开写，因为它们是两种不同的事实：

1. **捕获 vs padded eager：逐位相等**（60 步，状态、档案、解码 bitset 全部相同）。
   这是「捕获没有改变语义」。
2. **padded vs 未 padded：不是逐位相等，但等价**。空槽权重是 0，加 0 不改变和；但 `spmm`
   对非零项的归约顺序随行长变化，所以两个算子差在最后几位（每层约 1e-7，端到端 logits
   约 1.7e-5）。60 步后档案里 66–178 个 node-slot（占 4800 的 1.4 %–3.7 %）不同，
   **解的大小在 4/4 张图上完全一致**。所以 padded 引擎才是逐位对照的基准，与未 padded
   的差别按「距离」报告而不是按「相等」断言。

## 2. 结果（1000 张作者测试图，seed 1017，单进程、GPU1 独占）

`outputs/v8_macro/padded_short/padded/padded_ref0.csv` 与
`outputs/v8_macro/padded_long/padded/padded_ref0.csv`（捕获一次，`capture_s` 只在
`summary.json` 里记一次）：

| 每图时钟 | 解（摊薄） | 步数 | ms/步 | load | 解（per-graph 捕获） | 差额 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 s | **19.4040** | 67.0 | 0.591 | 6.4 ms | 17.6720（0 步） | +1.732 |
| 0.10 s | **19.6890** | 178.4 | 0.443 | 6.2 ms | 17.9947（0 步） | +1.694 |
| 0.20 s | **19.8410** | 413.7 | 0.384 | 5.9 ms | 19.3970（49.8 步） | +0.444 |
| 0.50 s | **19.9450** | 1309.9 | 0.343 | 6.3 ms | 19.8403（402 步） | +0.105 |
| 1.00 s | **19.9830** | 2887.1 | 0.329 | 6.3 ms | 19.9333（1058 步） | +0.050 |

0.2 s 一档的 prelude 从 93.5 ms 掉到 5.9 ms，同样 200 ms 里步数从 49.8 涨到 413.7（×8.3）。
1.0 s 一档步数从 1058 涨到 2887（×2.7），说明两千步以后仍未完全饱和，但每步的边际价值已经
很小（+0.05）。

### 2.1 每 N 块解码一次（`--decode_every`）

短时钟上解码占 40–44 %（0.1 s 档 62 ms 解码 / 78 ms 搜索）。档案是单调的，所以「每 3 块
解码一次 + 收尾补一次」与「每块都解」在质量上不会更差 —— 实测：

| 时钟 | 每块解（steps / 解码 / 解） | 每 3 块解 | 差额 |
| --- | --- | --- | --- |
| 0.10 s | 178.4 步 / 62.1 ms / 19.6890 | 200.6 步 / 21.7 ms / **19.7120** | +0.023 |
| 0.20 s | 413.7 步 / 103.0 ms / 19.8410 | 447.7 步 / 34.8 ms / **19.8480** | +0.007 |

步数只涨 12–8 %（解码省下的时间被「步长本身仍受发射限制」吃掉一半），解几乎不动。
**这是「短时钟的瓶颈不是工程而是模型的收敛速度」的直接证据**：把占 40 % 的开销几乎砍光，
只换来 +0.02。

## 3. 与论文表格对齐

| 方法 | 每图时钟 | RB[200–300] SIZE↑ | 说明 |
| --- | --- | --- | --- |
| RLNN | 0.197 s | 19.52 | 本文的对照主目标 |
| **本文（摊薄捕获）** | **0.20 s** | **19.841** | 同钟 **+0.32** |
| RLSA | 0.070 s | 19.97 | 同论文里更强的那个求解器 |
| 本文（摊薄捕获，0.07 s 插值） | 0.07 s | ≈19.55 | 同钟 **−0.42** |
| 本文（摊薄捕获） | 1.0 s | 19.983 | 首次越过 19.97，但慢 14 倍 |

两条结论：

1. **对 RLNN：两轴同时赢第一次成立**。0.2 s 对 0.197 s 时钟相同，SIZE 19.841 对 19.52。
   此前（per-graph 捕获）同一时钟只有 19.3970，是落后 0.12 的。
2. **对 RLSA：仍然输**。它在 0.070 s 就有 19.97；我们在 0.1 s 只有 19.689，要到 0.5 s 的
   19.945 才接近、1.0 s 的 19.983 才越过。想让「两轴赢所有基线」成立，还需要在 0.07–0.2 s
   之间再挤出约 0.42 —— 而按上面的边际曲线，靠加步数做不到（0.2→1.0 s 的 7 倍步数只值
   +0.14）。


> 判读纪律：全部是单种子（1017）。三个种子的 CI 还没跑 —— 表格里任何一句「赢」在补 CI 前
> 只能算点估计。

## 4. 复现

```bash
# 闸门
GRAPHS=4 STEPS=60 bash train_scripts/v8_macro/check_padded.sh
# 短时钟（1000 图，单进程独占 GPU1，约 5 分钟）
SHARDS=1 BUDGETS="0.05 0.1 0.2" DECODERS=cached SEEDS_LIST=1017 \
  RESULT_DIR=outputs/v8_macro/padded_short bash train_scripts/v8_macro/padded.sh
# 长时钟（约 26 分钟）
SHARDS=1 BUDGETS="0.5 1.0" DECODERS=cached SEEDS_LIST=1017 \
  RESULT_DIR=outputs/v8_macro/padded_long bash train_scripts/v8_macro/padded.sh
```

必须单进程：4 个分片并发时同样的 0.05 s 档只走得到 0–5 步（GPU1 的 kernel 发射被四路
串行化，单块耗时从 4 ms 涨到 40 ms 量级），第一版就是这么跑废的。

## 5. 产物

- `outputs/v8_macro/check_padded/check/check_ref0.json`（闸门）
- `outputs/v8_macro/padded_short/padded/padded_ref0.{csv,json}`（0.05/0.1/0.2 s）
- `outputs/v8_macro/padded_long/padded/padded_ref0.{csv,json}`（0.5/1.0 s）
