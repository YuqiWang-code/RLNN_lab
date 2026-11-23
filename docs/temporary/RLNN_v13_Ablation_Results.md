# RLNN v13 消融与融合结果（t25）

现行 0.2 秒截止协议的融合 PAVE 开关对照见 [项目现状](../项目现状.md) 与 [配对统计](../../outputs/v11/fused_pave_02s_deadline/summary.json)。

一句话：v13/v14 的三组消融已逐图从 CSV 重算、单独入簿到 `docs/实验结果.xlsx` 的 `V13 融合引擎`（192 行 × 21 列）；0.2 s 三臂 C/A/B = 19.8857/19.9560/19.9907，融合臂 B+ = 20.0327，数值等价性 |Δ|≥3 = 0。

## 三组数字（`csv_row` 指 `outputs/v11/collect/v13/` 下 CSV 的行号，可直接跳去复核）
- **G1 0.2 s 三臂**（3 seed × 1000 图，池均值）：C `19.8857` / 130.1 步 / PAVE 29.2 ms `arm_summary.csv:29`；A `19.9560` / 182.7 步 `:30`；B `19.9907` / 232.6 步（pave=false）`:31`。
- **G1 配对**（图级 cluster 自助 5000 次，95% CI；W/T/L 按 3000 对）：A−C `+0.0703` [+0.0593,+0.0817] 218/2772/10 `comparisons.csv:2`；B−A `+0.0347` [+0.0273,+0.0423] 109/2886/5 `:3`；B−C `+0.1050` [+0.0920,+0.1187] 317/2674/9 `:4`。
- **G2 0.1 s PAVE on/off**：A′ `19.7773` / 85.8 步 `arm_summary.csv:33`；B′ `19.8507` / 109.2 步 `:34`；B′−A′ `+0.0733` [+0.0623,+0.0847] 246/2726/28 `comparisons.csv:5`；旧协议 C′（前导解码 on）`18.5540` / 21.6 步 `arm_summary.csv:26`（report_only，不入均值）。
- **G3 三方**：B `19.9907`（0.2 s）`arm_summary.csv:31`；B+ `20.0327` / 366.9 步 / 0.542 ms/步 `:32`；A0（1 s 现况）`20.0687` / 810 步 `:35`。
- **G3 配对**：B+−B `+0.0420` [+0.0323,+0.0520] 151/2823/26 `comparisons.csv:6`；B+−A0 `−0.0360` [−0.0443,−0.0280] 6/2880/114 `:7`（同图配对、跨预算，不作等价性判据）。
- **常数基线（未配对）**：B−RLSA@0.2s `+0.0287`、B+−RLSA `+0.0707` `comparisons.csv:8,9`；A0−KaMIS `−0.0333` `:14`；L3（1 s 融合）`20.0870` `arm_summary.csv:36`、−RLSA `+0.0030` `comparisons.csv:12`（L3 行 report_only）。
- **等价性（200 图 × 固定 130 步）**：compiled vs plain identical `194/200`、max|Δ| `1`、|Δ|≥2 `0`、**|Δ|≥3 `0`**（ids = []）；默认路径 plain vs t24 curve130 `200/200` 相同、max|Δ| `0` `equivalence_summary.csv:2-7`。

## 列含义（逐列；192 行 = 臂×seed 25 格×5 指标 125 + parity 2 格×4 = 8 + 池均值 8 格×5 = 40 + 配对/常数 13 + 等价性 6）
| 列 | 含义 |
|---|---|
| version | 固定 `v13`；V11 表用 `v11`，两表按此列分流 |
| run | 臂名（`C(lead on,PAVE on)` …）、池均值（同名）、对比名（`A - C` …）或 `parity-compiled-plain` |
| task | `v13-<block>`：G1-0.2s / G2-0.1s / G3-fused / G3-A0 / G3-fused-1s / G2-legacy / G3-parity |
| dataset | `RB200-300 n=1000`；等价性探针 `n=200` |
| split | 固定 `eval` |
| epoch | 空（无训练阶段） |
| step | 空；步数走 `metric=steps` 行（与 V11 表同约定） |
| metric | `search_size` / `final_size` / `steps` / `ms_per_step` / `pave_ms` / `delta_search_size` / `equiv_*` |
| value | 臂行 = 1000 图均值；配对行 = 逐图差的池均值；等价性行 = 该检查的计数或逐图最大 \|Δ\| |
| unit | `indep_nodes` / `steps` / `ms` / `graphs` |
| source | 原始证据：`<run 目录>/v11_rows.csv` 或 `outputs/v11/collect/v13/comparisons.csv` |
| recorded_at | 入簿时间 |
| notes | 紧凑 JSON：budget_s / seed / n_graphs / mean_steps / pave / skip_incumbent / compile_model / 口径注明（B 臂带 search-only caveat） |
| ci_low | search_size 行与配对行的 95% CI 下界（图级自助 5000 次） |
| ci_high | 同上，上界 |
| win | 配对对比中 A 优于 B 的图数（阈值 1e-9）；未配对常数为空 |
| tie | 差为零的图数 |
| loss | A 差于 B 的图数；`win+tie+loss` = 对数（3000 = 3 seed × 1000 图） |
| n_graphs | 该行覆盖的图数（1000 / 200；配对行亦为 1000） |
| report_only | True = 记录用（C′、L3、parity 两臂、未配对常数），不进任何均值表 |
| csv_row | 本行数值在 `outputs/v11/collect/v13/*.csv` 的行号（复核入口） |

## 与 t9 的 V11 表如何区分：V11 表（`V11 C4-FIFO主线`）记 v11 主线的既有引擎（A0/P0/时钟/误配/curve/ladder）与 C4 FIFO 判据；本表只收 v13/v14 的调度层消融（关前导解码、关 PAVE、torch.compile 融合）与 200 图等价性探针。两表按 `version`（v11 / v13）与 `task` 前缀（v11-* / v13-*）分流，不共享任何均值行；A0 在两表同值（20.0687 / 20.0690），本表只把它当 G3 的 1 s 参照，不重复记账。
## 校验：34 个旧 sheet 维度不变（openpyxl + pandas 双验，0 moved）；192 行全部按 `csv_row` 回读一致（trace 192/192）；与 t21 的 `outputs/v11/r2/p05/*.json` 对账 9/9 PASS（max|Δ| = 3.3e-4，A0 差来自对方取 seed 1017 单点 20.069、本表池均值 20.0687）；快照 `docs/temporary/v13_20261118_20261118_085837.txt`（4.63 MB，334 个源文件块 + 两本工作簿共 36 个 worksheet，含 `V13 融合引擎`）。
