# R2 · 0.2 s 档 SOTA 审计（2026-11-17，纯盘上数据）

**结论（先行）**：**0.2 s 档不能写 SOTA。** 同卡（RTX 5090）、同 1000 图（`idx_0…idx_999`）、同档的 RLSA 实测是 **19.9620 @ 0.2010 s**（`docs/temporary/RLNN_v8_SOTA_Direction.md:30`）；我们 19.8410 @ 0.1880 s（`outputs/v8_macro/padded_short/padded/padded_ref0.csv`），**落后 0.121**。对仓库自带 RLSA run（19.9730 @ 0.2084 s）逐图配对差 **−0.132 [−0.160, −0.105]，34 胜 / 804 平 / 162 负**。唯一站得住的 0.2 s 表述是「对 RLNN 自家发布版 +0.32」（配对 +0.321 [+0.288, +0.355]）——**赢自家前作不是 SOTA**。

## A. 0.2 s 档实测清单（缺的一律写「未测」）

| 方法 | SIZE | 墙钟/图 | 实测还是推算 | 出处 |
| --- | ---: | ---: | --- | --- |
| RLSA `num_t=300`（num_t 扫描行） | **19.9620** | **0.2010 s**（中位 0.2000） | **实测**（同卡同图） | `RLNN_v8_SOTA_Direction.md:30`；等钟表 `:40` 同一格写 19.962 vs 19.8410、差 **−0.12** |
| RLSA 仓库基线 run（`num_t=300`） | 19.9730 | 0.2084 s（中位 0.2067） | **实测**，逐图 CSV 在盘 | `outputs/v1/rlsa/run.json`（`mean_size`/`mean_solve_time_s`）、`outputs/v1/rlsa/per_graph.csv`、`README.md:94` |
| RLNN 作者发布版（erdoes＋A.3 翻转） | 19.520 | **0.2081 s** | **实测**（同卡同图） | `outputs/v000_rlnn_baseline/rb_author_test_gpu1_author_erdoes_gradient_flip/per_graph.csv`、`README.md:141`（1.73 min/500 图） |
| RLNN 论文表 1（A6000） | 19.52 | 0.197 s | 论文值、异卡 | `README.md:96`、`:142`（1.64 min/500 图）、`:40`（口径换算） |
| KaMIS ReduMIS | **不可达** | 地板 1.87 s | 实测：`--time_limit=0.1` 仍 1.87 s 墙钟 ⇒ 0.2 s 内 **0/1000 图**返回 | `RLNN_v8_Positioning_Results.md:57`、`:65`、`:100`；`RLNN_v8_SOTA_Direction.md:53`（≤1 s 仅 99/1000、均值 19.8384）、`README.md:95` |
| 本文 0.2 s（摊薄捕获） | **19.8410**（final）/ 19.8340（search） | 0.1880 s，413.7 步 | 实测，**单 seed 1017**、`repeats=1` | `padded_ref0.csv`(budget=0.20)、`padded_ref0.json`；`README.md:75`；`RLNN_v8_Amortised_Capture.md:67`；`v10_paper_snapshot_A.txt:83` |
| 本文 0.2 s（逐图重捕获，3 seed） | 19.3970（final）/ 19.0947（search） | 0.1860 s，49.8 步 | 实测，seeds 1017/1019/1021 | `outputs/v8_macro/author_short/capture_budget/capture_budget_ref0.csv`(budget=0.20)；`RLNN_v8_Amortised_Capture.md:67` |

**插值交叉核对（不用于主结论）**：用 `num_t=1000` 20.0730@0.6505 s 与 `num_t=2000` 20.0930@1.2898 s（`RLNN_v8_SOTA_Direction.md:32-33`、`README.md:91-92`）做对数线性外推到 0.2 s 得 ≈20.039，**高于**实测 19.962 ⇒ 外推不可靠；0.2 s 这一档**不需要插值，盘上就有实测点**。别再引用 `RLNN_v8_Amortised_Capture.md:95` 的「RLSA 19.97 @ 0.070 s」——那个 0.070 s 是已作废的「整集合计/500」口径（`README.md:40`、`v10_paper_snapshot_A.txt:104`）。

## B. 判断：三选一

- ①「0.2 s 档最高（已实测对手范围内）」——**不成立**：RLSA 19.9620 实测就在 0.201 s。
- ②「与 RLSA 相当」——**不成立**：差 0.121–0.132 = 离散翻转分辨率 0.008 的 **15–16 倍**，不是「相当」。
- ③「0.2 s 档不可比」——**选它**，但理由必须改写：不是「对手无同预算实测」（有，而且我们落后），而是**同预算证据一旦成立，SOTA 主张即失效**，因此这一档只能作为「不可主张 SOTA」处理。
- **依赖假设**：RLSA 的 0.201 s 是固定 `num_t=300` 的实测墙钟，不是硬限时（`outputs/v1/rlsa/run.json` 里 `config.time_budget_s=null`）。若将来证明这一点在协议上不可比，③ 才会翻回 ①。
- **论文口径（唯一可写）**：「**两个预算档均不构成 SOTA**：1 s 落后 0.015（20.0690 vs 20.084），0.2 s 落后 0.121（19.8410 vs 19.9620@0.2010 s）。0.2 s 唯一成立的表述是对 RLNN +0.32（配对 +0.321 [+0.288,+0.355]，332 胜 / 645 平 / 23 负），不写 SOTA。」

## C. 要写进论文还缺什么（全部「未测」，不许估算）

1. **三 seed**：`padded_short` 只有 seed 1017、`repeats=1`（`padded_ref0.json`）⇒ 0.2 s 的 19.8410 没有 CI、没有 seed 方差；RLSA 侧同样单 seed 1017、无 repeat。
2. **同协议硬限时 RLSA**：`config.time_budget_s=null` ⇒ 「预算对齐」的对照不存在，现有只是「实测墙钟对齐」。要主张必须补 `--time_budget_s 0.2` 的 RLSA 实测。
3. **PAVE 计不计入**：19.8410 是 `final_size`（PAVE 仅 +0.0070，`pave_gain` 只有 0.7% 图 >0），`search_size` 是 19.8340；`solve_time_s=0.188` 与 `decode_time_s=0.103` 之和不闭合（列不可加），论文必须显式声明用哪个口径、decode/PAVE 在不在 0.188 s 内。
4. **KaMIS 缺席要写一句**：地板 1.87 s ⇒ 0/1000 图在 0.2 s 内出解，否则审稿人会当成漏测。
5. **scope**：只有 RB[200–300]；ER[700–800] 我们 1 s 42.18 输 KaMIS 44.00 / RLNN 43.34（`README.md:70`）⇒ 不能跨图族外推成「全域 SOTA」。

## D. 限制（写进正文，不能只当脚注）

- **0.2 s 的 19.8410 含「整集只捕获一次」的摊销**：`capture_s=0.817` 对 1000 图只付一次（摊 0.8 ms/图，`padded_ref0.json` 的 `padded.capture_s`）。**同一 0.2 s 若逐图重捕获只有 19.3970（49.8 步，3 seed 1017/1019/1021，`outputs/v8_macro/author_short/capture_budget/capture_budget_ref0.csv`），比 RLNN 19.520 还低 0.12**。审稿人必问「0.2 s 含不含捕获」，正面回答：捕获按整集摊销、不按图计。
- 19.8410 只有单 seed、无 CI；PAVE 口径不闭合（C.3）——两条按「未测」写，不估；t7 补三 seed。

原始数：`outputs/v11/r2/budget02_audit.json`（含逐图配对自助 5000 次、全部 file:line 引用）。
