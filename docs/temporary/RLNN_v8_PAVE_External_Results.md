# PAVE 外接判决：把 PAVE 接在 KaMIS 的解上

## 0. 要回答的问题

到 2026-11-12 为止，PAVE 唯一被验证过的场景是「接在我们自己的搜索输出上」，边际增益 +0.22（100 图，0.5 s 档）。
这个数字可以被两种完全不同的故事解释：

- 故事一（贡献版）：PAVE 是一个 **solver-agnostic** 的后处理，任何求解器的输出喂进去都能变好。若成立，论文标题可以是「a solver-agnostic post-hoc improvement for MIS」，并且可以在 KaMIS 的解上打出比 KaMIS 自己更好的数字。
- 故事二（平凡版）：PAVE 只是一个局部搜索，增益等于「输入离局部最优有多远」。若成立，KaMIS 的解（本身已经收敛）喂进去应当是 0，论文不能这么写。

判决实验：KaMIS `--time_limit=0.1` 的解 → A.3 → PAVE，看 20.045 变成多少。

## 1. 协议

- 100 张 validation 图（与所有 v7/v8 数字同口径），ref0 冻结骨干，冻结 PAVE 超参
  `outer_rounds=8, initial_rounds=3, horizon=2, candidate_cap=32, improving_cap=48, repair_rounds=4, anchored=False`。
- 每图一只给 0.1 s 的 PAVE 时钟（`--pave_limit 0.1`），KaMIS 也只给 0.1 s 算法时间。
- 臂：
  - `kamis`：KaMIS 原始解（基线）；
  - `kamis_extend`：只做一次 `SearchGraph.extend` 贪心扩展（不加任何学习，测「还有没有 1 步空间」）；
  - `kamis_pave`：PAVE + 学习 ranker（`PAVEV3.choose`）；
  - `kamis_pave_rule`：PAVE + 覆盖度规则（`choose="coverage"`，非学习对照）；
  - `kamis_pave_random`：PAVE + 均匀随机选（`random.Random(20260101)`，下界对照）；
  - `kamis_long`：KaMIS `--time_limit=0.2`，测「把 PAVE 那 0.1 s 直接还给 KaMIS 会不会更好」。
- 两条断言在每张图上强制：每个臂给出的集合必须独立（`is_independent`）；PAVE 结果不得小于输入
  （`platform_search` 只做 size+1 交换与单调扩展，构造上单调，断言用于防回归）。
- 图级配对自助（5000 次重采样，种子 20260101）给 CI。

## 2. 主结果：KaMIS 解上增益恰好为零

100 图，均值：

| 臂 | 解 | 相对 `kamis` | CI | 胜/平/负 |
| --- | --- | --- | --- | --- |
| `kamis`（0.1 s） | 20.0500 | — | — | — |
| `kamis_extend` | 20.0500 | +0.0000 | [+0.0000,+0.0000] | 0/100/0 |
| `kamis_pave` | 20.0500 | +0.0000 | [+0.0000,+0.0000] | 0/100/0 |
| `kamis_pave_rule` | 20.0500 | +0.0000 | [+0.0000,+0.0000] | 0/100/0 |
| `kamis_pave_random` | 20.0500 | +0.0000 | [+0.0000,+0.0000] | 0/100/0 |
| `kamis_long`（0.2 s） | 20.0500 | +0.0000 | [+0.0000,+0.0000] | 0/100/0 |

不是「小」，是**每一项都恰好是 0**，100 张图无一例外。这一批数字在独立重跑一次后逐位相同（`kamis_extend` 到 `kamis_long`
五个臂全部 `+0.0000`、`0/100/0`、均值同为 20.0500），所以不是单次运行的偶然。机制证据同一批行里：

- 三个 PAVE 臂的 `exchanges` 均值 0.00，且 **100/100 张图的交换次数都是 0**；
- `extend_gain` 均值 0.0——KaMIS 的解已经是极大独立集，没有可增的顶点；
- `deadline_hit` 只有 50/100（learned）、41/100（rule）、29/100（random），即 PAVE 大多在时钟用完之前就自己收敛退出了，不是「时间不够」；
- 单图墙钟：KaMIS 0.1 s 档 1.698 s（进程启动+写盘占 1.6 s 以上），PAVE 0.066 s。

## 3. 正对照：贪心解上 PAVE 有 +0.710

同一套代码、同一批图、同一个 0.1 s PAVE 时钟，只把起点换成 `minimum_degree` 贪心解：

| 臂 | 解 | 相对起点 | CI | 胜/平/负 |
| --- | --- | --- | --- | --- |
| `minimum_degree` | 18.3800 | — | — | — |
| `kamis_pave`（学习 ranker） | 19.0900 | +0.7100 | [+0.5600,+0.8700] | 53/47/0 |
| `kamis_pave_rule`（规则） | 18.9300 | +0.5500 | [+0.4300,+0.6800] | 47/53/0 |
| `kamis_pave_random`（随机） | 18.9500 | +0.5700 | [+0.4400,+0.7000] | 47/53/0 |

学习 ranker 相对两个非学习对照的优势：

| 对比 | 差 | CI | 胜/平/负 |
| --- | --- | --- | --- |
| learned − rule | +0.1600 | [+0.0800,+0.2500] | 15/84/1 |
| learned − random | +0.1400 | [+0.0700,+0.2200] | 14/85/1 |
| rule − random | −0.0200 | [−0.0800,+0.0300] | 2/95/3 |

即：+0.71 里的 **+0.55 来自「做局部搜索」这件事本身**（规则与随机都拿得到，两者互相之间 CI 跨零），
**+0.14~+0.16 来自学习到的选择器**，而且集中在 15/100 张图上。

## 4. 判读：PAVE 不是 solver-agnostic

把三次测量放在一条线上，增益是输入收敛程度的单调函数：

| 输入 | 输入大小 | PAVE 后 | 增益 | PAVE 时钟 |
| --- | --- | --- | --- | --- |
| 贪心最小度 | 18.3800 | 19.0900 | +0.710 | 0.1 s |
| 我们 eager 搜索 0.5 s | 19.3560 | 19.5780 | +0.222 | 0.1 s |
| KaMIS 0.1 s | 20.0500 | 20.0500 | +0.000 | 0.1 s |

故事二成立，故事一否掉：

- PAVE 的贡献不是「任何求解器的输出都能变好」，而是「把**还没有收敛**的解修到局部最优」。输入越接近局部最优，
  可拿的增益越少；KaMIS 的输出已经是 PAVE 搜索邻域的不动点，于是恰好为 0。
- 因此论文不能写「solver-agnostic post-hoc improvement」。写成那样，审稿人只要把 KaMIS 的解喂进来一次就推翻了。
- 唯一站得住的学习结论是 §3 最后那张表：**学习 ranker 相对规则/随机 +0.14~+0.16，CI 不含零**，
  但这条只在「起点很弱」时可见，且占总增益约 20%。

## 5. 对论文的影响

骨架 A（PAVE 一般化）判死。剩下两条：

- **B 吞吐量**：把搜索改成 batch GPU 吞吐，比的是「每秒能解多少图」。PAVE 在这里仍然有用（它把弱解修好），
  但结论会从「质量更好」变成「同样的墙钟里解更多图」，需要重新设计对照协议。
- **C 换 regime**：分布外图、大规模图、或者需要多解的场景。风险最高，收益也最不确定。

同时 §3 给出一个尚可用的正面素材：PAVE 能从任意弱构造器（这里是贪心，18.38）出发稳定拿到 +0.71，
这比「从我们自己的 0.5 s 搜索出发 +0.22」大 3 倍——说明它的价值集中在「弱起点」上。若走 B，这个性质正好用得上。

## 6. 复现

```bash
cd /home/hzeng/project/ZH/RLNN_lab
export RLNN_SOURCE_SHA=$(git rev-parse HEAD)     # 服务器无 git 时需要显式给

# 主实验：KaMIS 解（约 8 分钟，含 100 次 KaMIS 进程启动）
SOURCE=kamis SAMPLES=100 bash train_scripts/v8_macro/pave_external.sh

# 正对照：贪心起点（约 1 分钟）
SOURCE=min_degree SAMPLES=100 bash train_scripts/v8_macro/pave_external.sh
```

`SOURCE` 同时决定产物目录与 CSV 文件名，两个来源不会互相覆盖。

## 7. 产物

| 路径 | 内容 |
| --- | --- |
| `outputs/v8_macro/pave_external/pave_external_kamis_ref0.csv` | KaMIS 主实验逐图逐臂 100 行 |
| `outputs/v8_macro/pave_external_min_degree/pave_external_min_degree_ref0.csv` | 贪心正对照 100 行 |
| `outputs/v8_macro/queue/pave_external_kamis.log` | 主实验日志（含每 20 图进度与最终 delta 表） |
| `outputs/v8_macro/queue/pave_external_min_degree.log` | 正对照日志 |
| `models/eval_pave_external.py` | 实验脚本（`--source kamis|min_degree`） |
| `train_scripts/v8_macro/pave_external.sh` | 启动器 |
