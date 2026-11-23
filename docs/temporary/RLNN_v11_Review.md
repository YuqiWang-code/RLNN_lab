# RLNN v11 实现审查（t6）

> 审查对象：t5 交付（`models/mis_v11_unroll.py`、`models/probe_unroll_grad.py`、`models/mis_v11_gates.py`、
> `models/mis_v11_tabu.py`、`models/probe_tabu_walk.py`、`models/summarize_v11.py`、
> `train_scripts/v11/{a0_gate0,c3,probe_grad,smoke4}.sh` + README，以及 `models/{solvers,main,eval_v8_padded}.py`
> 的 5 处 additive 改动）与 `docs/temporary/RLNN_v11_Implementation.md`。
> 对照规格：`docs/temporary/RLNN_v11_Direction_Locked.md`（1058 行，§0–§14，§14.9/§14.10 为最新有效条款）。
> 审查方式：逐行读新文件 + 逐条比规格 + 本地实跑可离线部分 + 对服务器产物做自洽性核验。
> **未改任何代码**（本任务只审）。

结论：**不通过（needs_revision）**。引擎级与数值级交付我核到的部分全部为真（见 §3），
但有 **4 处判据级缺陷**（会让「未达判据」被记成「成立」或被硬阻断）与 **3 处不可复现/不实陈述**，
在它们修掉之前 t7 不应开跑——尤其 P1 会让 S1 判成假阳性，P4 会让 C4 的主线结论失去外部效度。

---

## 1. 七项检查表

| # | 检查项 | 结论 | 依据 |
| --- | --- | --- | --- |
| 1 | 实现与锁定方向逐条对应，有无偷换 | **部分不符** | C3 判死、A0 双口径、梯度探针判据、unroll 最后一次解码、K=200 固定部署：一致。C4 的**起点、预算、图数、ranker、h 网格全部偏离 §3.5**（P4）；C4 接线改为影子模块而非就地改冻结文件（P5，可接受但需断言）；`signal-only` 被当成功（P1）。 |
| 2 | 数据泄漏（测试集进训练/选型/早停） | **本地能查的部分通过，图级不重叠待证** | `models/solvers.py:426` 只用 `evaluate('valid')` 选模型；`models/main.py:113` 训练结束后才 `evaluate('test')`；`train_scripts/v11/c3.sh:50-52` 训练用 `RB_small_train`/`RB_small_val`，`--test_path` 只用于报告。**但**三份 split 的图级不重叠无法在本地验证（数据不在仓库，P13）；另有协议级复用（P12）。 |
| 3 | 计时与 deadline 语义（sync / 编译 / 加载归属） | **通过，两处口径须写清** | `engine.run` 每块 `sync_each_block=True`，`wall` 含同步；`--fixed_steps` 走 `cutoff=inf` + `max_blocks=ceil(808/10)=81` ⇒ 810 步，CSV `steps` 全 810；`load` 计入、`capture` 一次性不计入；拓扑构造在 `tick` 之前（各臂相同，不构成偏差）——**但**固定步数臂 PAVE 另取 `perf_counter()+reserve`，总墙钟≈1.24 s，不得对外称「1 s」（P9）。 |
| 4 | 基线可比性（同种子/同图/同时钟/同解码） | **机制通过，基线本身待裁** | A0 与 A1 走同一 `eval_v8_padded` 路径、同 `--seeds`、同图、同 `--fixed_steps`、同 `decode_top`；旧基线 `outputs/v8_macro/arm_fp16_dense` 与当前引擎差 +0.0075/+0.0080（P7，需 captain 裁）。 |
| 5 | 断言真断言还是被跳过（try/except 吞掉） | **发现 2 处恒真断言** | `models/mis_v11_gates.py:372-373` 的 `dtype_fp16`/`dense` 是把 CLI 字符串与自身比较（P2）；`models/mis_v11_gates.py:370` 用含 PAVE 抖动的 `final_size` 判 ±0.0005（P3）。全仓新文件**没有**用 `except: pass` 吞断言（grep `except` 仅 3 处，均为数据缺失时的显式跳过并打印）。 |
| 6 | 干净状态可复现（README 命令逐条） | **不通过** | 仓库工作树未提交（`git status` 五个冻结文件均有改动），`models/data/MIS_tensors/` 本地不存在，`summarize_v11` 在无 tqdm 的机器上直接 ImportError（P8/P11）。README 里的命令本身与脚本一致，但前置条件缺失。 |
| 7 | 性能回归（ms/步 是否使等钟不可比） | **通过** | 当前 1.287 ms/步 vs 旧基线 1.179 ms/步（+9%），固定步数口径下不可比性问题消失；1 s 时钟口径少 8.9% 步数，扣掉后同口径尺寸一致（20.0725 vs 20.0765 的差落在翻牌噪声内）。GPU 侧无新增图捕获成本。 |

---

## 2. 问题清单

按严重度排序；每条给「文件:行 → 问题 → 修法」。

### P1（高·判据放宽）`signal-only` 被算作 S1 成立
`models/summarize_v11.py:94`
```python
s1 = verdicts.get(1.0) in ('pass', 'signal-only') or verdicts.get(0.2) in ('pass', 'signal-only')
```
§2.7 / §14.9 明确：`+0.012~+0.020`（`signal-only`）**只进 Discussion，不算成功**。
现在这条会把「未达判据」直接印成 `S1 = True`，而 t7 的结论由这行决定。
**修法**：`('pass',)`；并把 `signal-only` 单独打印成「有信号未达判据」一行，供论文 Discussion 引用。
附带：`:86-88` 的优先级也应改为 `unstable > fail > signal-only > pass`，避免 `passed=True` 但稳健性不足时被归为 `signal-only`。

### P2（高·恒真断言）门 0 的 dtype/dense 自检没有读运行配置
`models/mis_v11_gates.py:372-373`
```python
checks['dtype_fp16'] = args.expect_dtype == 'fp16'
checks['dense'] = bool(args.expect_dense)
```
`train_scripts/v11/a0_gate0.sh:44` 恒定传 `--expect_dtype fp16 --expect_dense 1` ⇒ 两项恒为参考 true/true。
更糟的是这条脚本读的是**既有命名/既有 schema** 的 CSV（`train_scripts/v11/a0_gate0.sh:40` 自注「gate0 读的是
既有 schema 的 CSV（padded/padded_ref0.csv）」，实为本次 `$OUT/$NAME` 下新写出的那份、沿用旧文件名），
即这两个 flag 字面上只是「参考配置应该是 fp16+dense」，而**被测臂**是不是 fp16+dense 根本没查：`--v11_csv` 的 schema（`models/eval_v8_padded.py:377-393`）没有 dtype/dense 列，
summary JSON（`:326-334`）也没有。这正是 captain 第 3 条要防的「配置漂移」失效点。
**修法**（任选一，建议两者都做）：
1. 给 v11 CSV 每行加 `dtype`、`dense` 两列（`solver.dtype` 在 `models/eval_v8_padded.py:169`，
   `engine.use_dense` 在 `models/mis_v8_pad.py:80`，`--dense` 开关在 `models/eval_v8_padded.py:135`），
   门 0 从 CSV 聚合并要求 100% 一致；
2. `mis_v11_gates.py` 直接读 `run.json` 的 `mixed_precision` 与 engine 的 `dense` 构造参数。

### P3（中·自相矛盾）门 0 solution 判据用了含 PAVE 抖动的 `final_size`
`models/mis_v11_gates.py:370` 用 `mean_size`（= `final_size`）对 20.0685 判 ±0.0005，
而 `final_size` 含末端 PAVE，其墙钟抖动给 1000 图均值带来 ±0.0005 量级噪声（`:358`；t5 报告 §5.9 自己写明「≤0.001 比较必须用 `search_size` 或 `--reserve_frac 0`」）。判据因此不可达/不可判定。
**修法**：门 0 主判据改 `search_size`（旧基线 search=final=20.0685，无 PAVE 增益，见 `outputs/v8_macro/arm_fp16_dense` 的 `pave_gain=0`），`final_size` 只作报告。

### P4（高·外部效度）C4 探针跑在规格之外，AR 不可迁移
锁定 §3.5 的探针命令：**1000 图**、`--h 8,16,32`、`--budget 0.1`、`--rankers outputs/v3/stats_h1_seed0/checkpoints/best.pt`。
实跑（`train_scripts/v11/README.md:20`）：**200 图**、h 到 128、`--budget 0`、**无 rankers**。
`models/probe_tabu_walk.py:14` 注释写 `--rankers`「accepted for interface compatibility」并被忽略；
`:62-63` 的 `greedy_mis()` docstring 声称贪心解是「the input a converged search starts from」——
实测该起点 17.1050，而部署解约 20.0（差近 3 点），前提是错的。
**后果**：`AR=0.988`、`ER=0` 是「从 17.1 的未收敛解起步、无 PAVE、0.1 s→0 s」条件下的数，
**不能**与锁定文档 §3.4 的「当前 AR=0」并列，也不满足「1 s 档上成立」的判据要求。
`ER=0` 本身我另有独立佐证（见 §3 的等价性实验），但那是机制结论，不是过门证据。
**修法**：按 §3.5 原命令重跑一次（GPU1 ≈10 min），或由 captain 明确改写 §3.5 并写明「AR 只在未收敛输入上有意义」。

### P5（中·可维护性）C4 走影子实现，缺等价性断言
§3.3 要求就地改 `models/mis_v3_search.py:197/247/340`；实现改为 `models/mis_v11_tabu.py` 内的
`action_key`(:30) / `improving_actions_tabu`(:35) / `repair_tabu`(:59) / `pool_plain_and_tabu`(:101) /
`platform_search_tabu`(:117) / `rollout_tabu`(:200)（`train_scripts/v11/README.md:53-54` 明说不改冻结文件）。
影子实现会让「冻结平台」与「C4 平台」将来分叉且无人发现。
我用合成 ER 图（n=120、p=0.15、greedy 起点、`outer_rounds=3`、seed 1017/1019）验证：
`fifo=None` 时 `best`/`work` 与 `platform_search` **逐位相同**，`h=8/32/128` 亦全同。
**修法**：把这条等价性写成 `models/probe_tabu_walk.py` 的一个自检断言（`fifo=None` ⇒ 与 `platform_search` 逐位相同），
每次跑 C4 前先过它；文档里把「未改冻结文件」的理由从「避免污染」改成「有等价性断言保护」。

### P6（中·不实陈述）「逐字节未变 / git diff 可证」三处站不住
- `docs/temporary/RLNN_v11_Implementation.md:271` 称冻结协议「零改动（`git diff` 可证）」——
  实际 `git status`：`models/mis_v8_pad.py`(+75)、`models/models.py`(+19)、`models/solvers.py`(+11)、
  `models/eval_v8_padded.py`(+176)、`models/main.py`(+19) 全部未提交，且 v9（dense/scale）、v10（keep/tabu）、
  v11（unroll/fixed_steps）混在同一批改动里，**git 无法区分**。
- `docs/temporary/RLNN_v11_Implementation.md:174` 与 `train_scripts/v11/README.md:50` 称既有 CSV「字段未改/逐字节不变」——
  既有逐行 dict 新增了 `recombine_size/recombine_pool/recombine_s/lns_rounds/lns_evicted`
  （`models/eval_v8_padded.py:298-302`），新写出的表头必然不同（按列名读的消费者不受影响，t5 在 `:174` 也承认）。
**修法**：改成「v11 只新增 X；Y 的改动来自 v9/v10（可查 mtime 与内容 diff），本任务未再动」，
并删掉「git diff 可证」这类无法执行的证明承诺。

### P7（高·需 captain 裁）A0 基线在当前引擎上不可复现，且脚本会硬阻断 t7
当前引擎同配方（K=200、dense+fp16、810 步、seed 1017、1000 图）得 **search 20.0760 / final 20.0765**，
与三周前 20.0685 差 +0.0075/+0.0080，`solution=false`。
t5 的归因（旧 CSV 写盘 2026-11-16 14:09 早于 `models/mis_v8_pad.py` 的 mtime 20:51 6.7 h）与我自查一致；
`run1/run2` 逐图 1000/1000 相同也说明现引擎确定。但 `train_scripts/v11/a0_gate0.sh:71` `exit $GATE_I_RC`
按 ±0.0005 判死，**t7 按 README 顺序（第 3 步不过不许开第 4 步）会被永久阻断**。
**需 captain 选**：
(a) 保留 20.0685 但把门 0(i) 容差改为 `|Δ| ≤ 0.01` 且注明「翻牌噪声底 0.008」；
(b) 采纳 t5 建议，把 **810 步 20.0760（search）** 记为 A0 步数口径基线，20.0685 降为历史时钟参照，
并在论文里同时报两者与差值来源。
我倾向 (b)：门 0 的目的是「证明协议没漂」，不是「证明三周前那个数字」。

### P8（高·交接风险）干净状态不可复现
`git status` 全部改动未提交；`models/data/MIS_tensors/` 本地不存在（只有 `models/data/*_generator.py`）；
`train_scripts/v11/README.md:3` 的「前置」只写「服务器上两个 checkpoint 存在」。
新克隆 / 新同学无法复现任何门。
**修法**：出一个 `docs/temporary/RLNN_v11_Code_Snapshot.md`（或直接 commit + tag），列明
① 涉及文件清单与改动摘要 ② 数据目录清单与生成种子 ③ 两个 checkpoint 的 sha256/字节数（668801 B、18225 B）。

### P9（低-中·对外口径）固定步数臂不是「1 s」
`models/eval_v8_padded.py:264-265`：`pave_deadline = perf_counter() + reserve`，reserve 默认 0.2 s
⇒ 810 步臂总墙钟 ≈ 810×1.287 ms + load 6.7 ms + PAVE ≤0.2 s ≈ **1.25 s**。
t5 报告已注明，但摘要与后续快照不得写成「1 s 档」。
**修法**：CSV/摘要统一标 `budget=fixed808`，或加 `wall_total_s` 列。

### P10（低-中·健壮性）`models/solvers.py:21` 顶层硬 import
`from mis_v11_unroll import unroll_graph_loss` 是无条件顶层 import，而同文件 `:277-283` 的
`mis_v8_global`/`mis_v9_scale` 都是惰性 import ⇒ 删/改名该文件会让**整个仓库**无法 import。
**修法**：移进 `elif self.loss == 'unroll'` 分支。

### P11（低·可移植）`models/summarize_v11.py:26` 依赖整条训练栈
`from eval_v8_capture_budget import paired_bootstrap` → `diagnose_v7_census` → `eval_v2` → `main` → `solvers`
→ tqdm/torch_geometric/accelerate。本机实测 `ModuleNotFoundError: No module named 'tqdm'`。
纯统计脚本在评审机/CI 上跑不起来，等于「结论只能在一台机器上算」。
**修法**：把 `paired_bootstrap`（`models/eval_v8_capture_budget.py:91`）复制进 `models/mis_v11_stats.py`（无重依赖）。

### P12（中·论文层）协议级测试集复用
v8–v11 的所有配置（K、dense+fp16、decode_top、unroll 长度、PAVE 预算）都在同一 1000 张
`RB_small_author_test` 上选过，无独立 holdout；v11 的 C4 过门也在这 1000 图上。
**修法**：论文显式披露；或用 `RB_small_val`(100 图) 之外、从未参与调参的子集（如作者集的奇数索引）做一次复核（≈0.3 GPU·h）。

### P13（待证·数据泄漏）split 图级不重叠未验证
`train_scripts/v11/c3.sh:50-52` 用 `RB_small_train` / `RB_small_val`，但本地无数据，无法核对三份 split 的图级重叠。
**修法**：t7 在服务器跑一次 `sha256`/边集哈希清单比对，把结果写进 `outputs/v11/`（≈2 min）。

---

## 3. 我独立验证过、结论为正的部分（供 captain 校准本次审查）

1. **可编译**：六个新文件 `python -m py_compile` 全过。
2. **CPU selftest 9/9 本地复现**：`python models/mis_v11_gates.py selftest` 在本机通过并写出
   `outputs/v11/gates_selftest.json`（finite / nonzero_grad / feasible / decode_matches_reference /
   decode_from_feasible_stays_feasible / archive_reported / same_seed_bitwise / degrade_recorded / degrade_trace）。
   注：本机走 `models/mis_v11_gates.py:68-90` 的 `_Copy` 回退（无 torch_geometric），
   即「reference」是 A.3 的内联副本；GPU engine 闸门 7/7 才用真 `problems.mis.Problem`——这一点应在文档里写明。
3. **影子平台等价**：`platform_search_tabu(..., fifo=None)` 与 `platform_search` 逐位相同（2 个合成图、2 个种子）；
   `h=8/32/128` 与 plain 全同 ⇒ **ER=0 与起点无关**，是 FIFO 键空间（10⁴–10⁵）远大于 h 的结构性结果。
4. **`--fixed_steps` 语义正确**：`cutoff=inf`、`max_blocks=ceil(808/10)=81` ⇒ 810 步；CSV `steps` 全 810。
5. **训练期解码口径合规**：`models/mis_v11_unroll.py:170-171` 取 `out['last_size']`（每 10 步解码的**最后一次**），
   符合 captain 第 2 条；`archive=True` 才取 archive 均值。
6. **梯度探针判据是预注册判据**：20 图的「pass」完全按 §2.9 字面（Δ(+)<0、ρ>0.2、grad:control≥1）判定，
   实测 −0.0125 / 0.224 / 1.067 恰好过线，CI 全跨 0 也如实报了——这不是事后改判据。
7. **统计三条款实现正确**：pooled 自助 CI（`models/summarize_v11.py:60`）、`≥2/3` seed 同号（`:85`）、
   任一 seed < −0.010 判 unstable（`:63`）——除 P1 的 S1 放宽外无问题。
8. **无泄漏的判定顺序**：`models/solvers.py:426` 只用 valid 选模型，`models/main.py:113` 训练后才测 test。
9. **新文件里没有吞断言**：`except` 仅 3 处，均为显式数据缺失跳过并打印原因。

---

## 4. 给 captain 的裁决请求

1. **P7 选 (a) 还是 (b)**：门 0 用 20.0685±0.01，还是改用本次 810 步 20.0760（search）作 A0 基线、20.0685 降历史参照？
   （不裁则 `a0_gate0.sh` 会硬阻断 t7。）
2. **P4 重跑还是改规格**：C4 探针按 §3.5 原命令（1000 图 / h=8,16,32 / budget 0.1 / 带 ranker）重跑 10 min，
   还是把 §3.5 改成「200 图、未收敛起点、budget 0」并声明 AR 只作机制量、不作过门判据？
3. **P1/P2/P3 是否要求 t5 先修再开 t7**：这三条都是判据级，且都在 CPU 可改（<1 h）、改完需重跑门 0 一次（≈0.4 GPU·h）。
   我建议：**先修 P1–P3 + 加 P5 的等价断言，再重跑门 0，然后才开 t7**。

---

## 5. 审查元信息

- 本地可跑：`py_compile`、`mis_v11_gates.py selftest`、合成图上的平台等价性、`git status`/`git diff --stat`、静态阅读全部新文件与 5 处冻结改动。
- 本地不可跑（无 GPU/无数据/无 tqdm）：engine 7/7、门 0 复算、C4 探针、`summarize_v11`。
  这几项我按「产物自洽性 + 代码逻辑」审，未按「独立复算」审，结论已在对应条目注明依据。
- 审查未发现「偷偷改判据方向」的行为：所有偏离都是**放宽或跑偏**，且大多在 t5 报告里自曝（§5.9/§7.1/§9），
  唯一未自曝的是 P1 与 P2。

---

## 6. t13 复审（2026-11-17 22:20–23:0x）：判据层还有没有假阳性通道

**复审对象**：t11 对 R1–R7 的修复（`models/mis_v11_gates.py`、`models/summarize_v11.py`、
`models/eval_v8_padded.py`、`train_scripts/v11/a0_gate0.sh`、`RLNN_v11_Implementation.md`、`README.md`）。
**复审口径**：只看「判据能不能被判据本身骗过」，不重跑结论；本机 CPU 能跑的一律实跑，
服务器项按产物 + 代码逻辑审。**未改动任何代码**（下面给的修法由 implementer 执行）。

### 6.1 一句话结论

**判据层的三个大洞（signal-only、恒真 dtype、final_size 判据）已堵住，负例机制也真的能失败；
但 A0 定稿改动（20.0760 → 20.0690）只落到了 README 与 Implementation 的开头，没有落到判据常量、
JSON 字段和 §7.4/§7.5 正文，导致「门 0 拿 fp32 误配值当参考」这一条仍在。** 判为 **needs_revision**。

### 6.2 逐条销账（R1–R7 + captain 新增 A0 三条）

| 编号 | 复审项 | 结论 | 证据（文件:行 / 实测） |
| --- | --- | --- | --- |
| **R1** | signal-only 在 S1 与 passed 两条路径上被拦住；优先级 `unstable > fail > signal-only > pass` | **通过** | `models/summarize_v11.py:89-96` 固定优先级；`:108` `s1 = any(verdicts.get(b)=='pass' ...)` 只认 pass；`:102-107` signal-only 单独打印「Discussion only」。**触发样例（本地实跑）**：合成 3 seed × 30 图，seed1017 +0.12、1019/1021 −0.001 ⇒ `Δ +0.0393 CI [+0.0286,+0.0514]`、`+seeds 1/3` ⇒ verdict=`signal-only`、`S1 = False`。注：该脚本没有第二个 `passed` 聚合字段，S1 是唯一总闸——所谓「两条路径」实际只有一条，已如实写明。 |
| **R2** | dtype/dense 取自被测 run 的 CSV，而不是 flag 自指 | **通过** | `models/eval_v8_padded.py:383-386` `dtype_tag` 由 `solver.dtype` 映射、`dense_flag = int(bool(getattr(engine,'use_dense',args.dense)))`、`fp32_csr = int(dtype_tag!='fp16' and not dense_flag)`；`:404-405` 逐行写入 v11 CSV；`models/mis_v11_gates.py:472-484` 五项检查全部读 `artifact[...]`，`:484` 要求 `artifact_consistent`（100% 行一致）。`--expect_*` 已从 gates 与 `train_scripts/v11/a0_gate0.sh` 中消失（`grep` 命中 0）。**反证**：fp32 误配跑被 `dtype_fp16=false` 判 exit 3（`Implementation.md` §7.6）。 |
| **R3** | 门 0 只用 search_size + \|Δ\|≤0.01 + 810 步硬校验；JSON 有 noise_floor；20.0685 标 historical | **部分通过，A0 未落地** | search_size 判据 `models/mis_v11_gates.py:476`；步数 `:480`；historical 字段 `:493-495` 齐全。**但** `:66 REFERENCE_SEARCH = 20.0760` 是 fp32 误配跑的值，不是 A0 定稿 20.0690；`:67 NOISE_FLOOR = 0.008` 仍在、`:492` 仍写进 JSON；`:489 reference_kind='A0_current_build_810_steps'` 与新定稿名不符（见 F1）。 |
| **R4** | C4 结案措辞收窄 + 三点证据 | **通过（文档层）** | `docs/temporary/RLNN_v11_Implementation.md:215-216`「在同一移动族且起点已极大时，记忆只能重排/收缩候选、不能创造 +1 移动；M4（k≤6 精确 648 格全 0%）+ 三重不动点 ⇒ 增益上界 0」；三点证据（机制 ER=0.0668/命中 4493、收益 gain +1.800→+1.745、起点不可迁移）齐备；`README.md:38` 同步。方向文档 §14.10.4 仍是旧规格（AR≥1/8 作决策锚），已按授权在 §14.10.9 **追加**措辞更正（不改正文）。 |
| **R5** | 门 0 能失败（负例 exit≠0） | **通过（并已在新 A0 下复验）** | 本地 `python models/mis_v11_gates.py gate0 --negative-selftest` ⇒ `clean_passes/shift_fails/config_fails` 全 true、exit 0（该模式「验证负例失败」本身返回 0 是正确的；判死逻辑在 `:564-566`：shift 必须 `checks['search'] is False`，fp32 必须三项同 false）。**关键复验（captain 新增第 4 条）**：把 `REFERENCE_SEARCH` 临时改为 20.0690 后再跑，`shift` 仍被 search 判死、`fp32` 仍被 `dtype_fp16/dense/not_fp32_csr` 三项判死 ⇒ **改数不会让负例失效**。 |
| **R6** | 「逐字节未变 / git diff 可证」按事实改写 | **未通过（残留）** | 事实表已进 `Implementation.md:423-430`（五文件未提交、`models/data/` 不入库、无 ckpt 快照、旧 run 不可考）——这部分对。但全文仍有「**逐字节未变**」`Implementation.md:11`、「**逐字节不变**」`train_scripts/v11/README.md:50`、「**逐字节等价**」`Implementation.md:59`（见 F3）。 |
| **R7** | 干净状态可复现 + 脚本不写死期望 | **部分通过** | `train_scripts/v11/a0_gate0.sh` 已无 `--expect_*`，判据只剩 gate0 常量 + `--expect_graphs 1000`；但 `:6` 注释仍写「-> mean search_size **20.0760** ± 0.01（现行构建的 A0）」（见 F2）。干净状态另一条（data 不入库/无 ckpt）已在 §7.4 明示，属可接受限制。 |
| **A0-1** | 全文是否还有 20.0765/20.0760 被当 A0 | **未通过** | 被当 A0/参考值：`models/mis_v11_gates.py:26,66,489`；`train_scripts/v11/a0_gate0.sh:6`；`Implementation.md:14(尾句),236,250,543,555`；`docs/temporary/RLNN_v12_Round2_Directive.md:9`；`docs/temporary/RLNN_v12_SecondContribution.md:54,75`。应只剩「fp32 误配」一处（`Implementation.md:247,529-530` 与 §7.6 表格那两行）。 |
| **A0-2** | 「噪声底」「离散翻转分辨率」是否清零、统一为「配置敏感度 0.007」 | **未通过** | 两词仍在：`README.md:36`、`Implementation.md:14,270,355,375,403,409,445-446,534,540`、`a0_gate0.sh` 无但 gates 仍 `NOISE_FLOOR`。t11 自己在 `Implementation.md:534` 已写「0.007 = fp32 与 fp16 的数值路径差」，但没有把术语统一过去。 |
| **A0-3** | README 与 Implementation §7.4/§7.5/§7.6 口径一致 | **未通过** | `README.md:32` 对 RLSA 写「**都只能写「打平」**」（与 captain 2026-11-17 22:19 定稿「写落后 0.015、不许写打平」冲突）；`README.md:28` 仍留 KaMIS **1 s 20.1060**（审计口径应为 20.1020@3.046 s）；`§7.4` 标题仍把 27 张翻牌定因为「解码时刻/构建」，而 `§7.6` 已定因为 **fp32 vs fp16**，两节互相矛盾；`§7.5` 表头/来源行仍写「旧 vs 新 `a0_steps808_n1000`」，未改成「fp32 误配 vs 声明配置」。 |

### 6.3 问题清单（needs_revision 的最小集，按严重度）

**F1（high）门 0 的 A0 参考值与 JSON 字段仍是 fp32 误配口径**
- `models/mis_v11_gates.py:66` `REFERENCE_SEARCH = 20.0760` —— A0 定稿是 **20.0690**（声明配置 fp16+dense、810 步）。
- `models/mis_v11_gates.py:67` `NOISE_FLOOR = 0.008` 与 `:492 noise_floor=NOISE_FLOOR` —— captain 要求删掉，改为 **配置敏感度 0.007**（fp16 vs fp32 的路径差），它不是噪声底。
- `models/mis_v11_gates.py:489` `reference_kind='A0_current_build_810_steps'` —— 应为 `A0_declared_config_fp16_dense_810_steps`，并同时记 `reference=20.0690`。
- **为什么是判据缺陷而不是措辞**：参考值偏 +0.007 而容差 ±0.01 ⇒ 验收带变成 [20.0660, 20.0860]；正确带是 [20.0590, 20.0790]。**20.080–20.086 这段退化（例如丢掉 dense/fp16 的一半收益）会被当前门判 pass**，这正是「判据层的假阳性通道」。
- **修法**：`REFERENCE_SEARCH = 20.0690`；删 `NOISE_FLOOR`，新增 `CONFIG_SENSITIVITY = 0.007` 并写进 JSON（`configuration_sensitivity`）；`reference_kind` 改新名；`negative_selftest` 的 clean 档自动跟随常量（`:557` 已如此），无需改逻辑。
- **已验证安全**：改数后负例仍 fail（见 §6.2 R5）。

**F2（high）§7.4/§7.5 与 §7.6 自相矛盾，20.0760 仍被当 A0/主语**
- `docs/temporary/RLNN_v11_Implementation.md:409` 标题「离散翻转分辨率 0.008：27 张 ±1 翻牌的定因」把翻牌归因于解码时刻/构建；`:452` §7.5 表头「27 张 ±1」、`:453-454` 来源行「旧 vs 新 `a0_steps808_n1000`」；而 `:505-514` §7.6 已证明 27 张翻牌来自 **fp32 误配**，声明配置只有 1 张（`graph_000466`）。
- `:236,:250,:543,:555` 仍把 20.0760 记为 A0 参考；`:14` 尾句「A0 参考仍用 20.0760」。
- `train_scripts/v11/a0_gate0.sh:6` 注释同上。
- **修法**：§7.4 标题与定因改「27 张翻牌 = fp32 误配 vs 声明配置的数值路径差（0.007），与解码时刻无关的对照见 §7.6」；§7.5 表头改「fp32 误配 vs 声明配置」，来源行明确哪份 CSV 是 fp32 跑；所有「A0 参考 20.0760」改 20.0690，并把 20.0760 限定在「fp32 误配」语境。

**F3（medium）「逐字节」残留三处**
- `docs/temporary/RLNN_v11_Implementation.md:11`「每条既有代码路径逐字节未变」；`:59`「逐字节等价」；`train_scripts/v11/README.md:50`「既有 CSV/JSON/summary 逐字节不变」。
- **修法**：改成「v11 只新增 X；Y 的改动来自 v9/v10，无法用 git 区分（§7.4）；既有 CSV 新增 5 列，按列名读取的消费者不受影响」。

**F4（medium）术语未清零**
- `README.md:36`、`Implementation.md:14,270,355,375,403,409,445-446,534,540` 仍用「离散翻转分辨率 / 噪声底」。
- **修法**：统一为「**配置敏感度 0.007**（fp16 vs fp32 数值路径差；同配置两次运行逐位相同，抖动 0）」，只在引用历史结论时保留旧词并注明。

**F5（medium）对外口径与 captain 定稿冲突**
- `README.md:32` 对 RLSA 写「只能写打平」；captain 22:19 定稿：**写「落后 0.015」，不许写「打平」**。
- `README.md:28` 仍留 KaMIS「1 s 20.1060」（审计口径 20.1020@3.046 s，另有 ≤1 s 只有 99/1000 图返回、那 99 图均值 19.8384 与 1.87 s 墙钟地板两个事实必须同时写）。
- **修法**：按定稿改字；KaMIS 行替换为 20.1020@3.046 s + 两个可审计事实。

**F6（low，非本轮引入）`summarize_v11.py` 仍依赖整条训练栈**
- `models/summarize_v11.py:26 from eval_v8_capture_budget import paired_bootstrap` ⇒ 本机实跑 `ModuleNotFoundError: No module named 'tqdm'`（经 `eval_v2 → main → solvers`）。判据不受影响（GPU 机有 tqdm），但「纯统计脚本不能单独跑」仍是可移植性缺陷；复审时我用 stub 注入 `paired_bootstrap` 完成 R1 的触发测试。

### 6.4 复审用到的实测命令（可复现）

```bash
python models/mis_v11_gates.py selftest                       # 10/10, passed=true, exit 0
python models/mis_v11_gates.py gate0 --negative-selftest      # clean/shift/config 全 true, exit 0
python - <<'PY'                                               # 新 A0 下负例仍 fail
import mis_v11_gates as g; g.REFERENCE_SEARCH = 20.0690
... g.negative_selftest(...)  -> shift.search False, fp32 三项 False
PY
python - <<'PY'                                               # signal-only 触发样例（stub 掉 paired_bootstrap）
... S.main() -> verdict signal-only, S1 False
PY
```

### 6.5 复审元信息

- 本地实跑：`selftest`（10/10）、`gate0 --negative-selftest`（3/3）、`REFERENCE_SEARCH=20.0690` 下的负例复验、
  `summarize_v11` 的 signal-only 触发样例、`py_compile`、`grep` 全仓一致性。
- 本地不可跑：服务器侧 `gates_gate0.json`（`outputs/v11/gate0/` 未同步到本地）、门 0 复算。F1 的结论不依赖服务器产物——
  它是常量/字段级事实，读代码即可判定。
- 复审未发现「故意把判据改宽」的行为：五处修复方向都正确，只是 A0 定稿晚于 t11 的落地时间，
  差在「常量与正文没跟上」，不是设计错误。

## 7. t13 复评（attempt 2，2026-11-17 23:40–）：t16 的 A0 对齐验收

**复评范围（captain 限定）**：只查 F1–F5 + 门 0 负例；**不重开 R1–R7**（§6 已销账的部分不重复审）。
**复审对象**：`models/mis_v11_gates.py`（22:36）、`README.md`（23:15）、`docs/temporary/RLNN_v11_Implementation.md`（23:39）、`train_scripts/v11/{a0_gate0.sh,README.md}`。
**方式**：`grep` 全仓一致性 + 逐行读改动 + 本地实跑负例。**未改任何代码**（本任务只审；§14.10.9 的追加除外，已在上轮完成）。

### 7.1 一句话结论

**F1/F3/F4/F5 与门 0 负例都已真修好（常量、JSON 字段、README 口径、负例复跑全部对齐）；只剩 F2 的一半：§7.4 与 §7.3 正文仍把 27 张翻牌定因为「解码时刻」，与 §7.6 的 fp32↔fp16 定因直接冲突。** 判 **needs_revision（1 medium + 1 low）**——不是判据缺陷，是同一份产物里两个互斥的因果句，会被论文直接引用。

### 7.2 逐条销账

| 编号 | 复评项 | 结论 | 证据（文件:行 / 实测） |
| --- | --- | --- | --- |
| **F1** | `REFERENCE_SEARCH` 改 20.0690、删 `NOISE_FLOOR`、`reference_kind` 改名、容差不含 0.007 | **通过** | `models/mis_v11_gates.py:75 REFERENCE_SEARCH = 20.0690`（注释「A0 = declared config (fp16+dense), 810 steps」）、`:76 REFERENCE_SEARCH_TOLERANCE = 0.01`、`:77 CONFIG_SENSITIVITY = 0.007`（注释「narration only, NOT in the tolerance」）、`:485 search=abs(mean_search - REFERENCE_SEARCH) <= args.tolerance`、`:498-502 reference/reference_kind='A0_declared_config_fp16_dense_810_steps'/delta_search/config_sensitivity`；`grep -n "NOISE_FLOOR\|noise_floor" models/mis_v11_gates.py` **命中 0**。验收带 = **[20.0590, 20.0790]**（正是原 F1 要求的正确带；此前 20.080–20.086 的假通过区间已被排除）。合成 clean 档随之更新（`:535-536`），负例沿用同一常量（`:568`）。 |
| **F2** | §7.4/§7.5 口径改「fp32 误配 vs 声明配置」、与 §7.6 一致 | **部分通过（正文冲突残留）** | 标题已改：`Implementation.md:418`「27 张 ±1 翻牌的定因：**fp32 误配 vs 声明配置**（HEAD 构建实测 + 仓库事实）」、`:462`「### 7.5 **fp32 误配 vs 声明配置**的翻牌台账」、`:464`「本节的 27 张属于 fp32 误配跑」。20.0760 已从所有「A0/参考值」位置撤掉：`:14`（明确「A0 参考 = 20.0690」「20.0760 是 fp32 误配」）、`:247`（表格行标注「(i)′ fp32+dense：**配置偏差**」）、`:250`、`:315`（「只作 `misconfig_fp32_sensitivity = 0.007` 的证据」）、`:543`、`:557`；`train_scripts/v11/a0_gate0.sh:6` 已改为「20.0690 ± 0.01（声明配置）… 20.0760 是 fp32 误配，只作 … 证据」。**但正文仍有互斥因果句**（见 F2-R）：`:401-402`「真正成因是**解码时刻的 pad / 浮点累加路径差**」与 `:450-452`「**定因**：… 差异不在引擎，而在**解码时刻**」，而 `:455-460` 与 §7.6（`:528-560`）把同一批 27 张归因为 **fp32↔fp16 数值路径差**。 |
| **F3** | 删「逐字节未变」类表述、改写为事实表 | **通过（1 处引述残留）** | `Implementation.md:11` 现为「每条既有代码路径未改动（既有代码的**未提交状态**见 §7.4——`git diff` 不能证明零改动；门 0 用同一份代码复现基线来证明行为等价）」；`:59` 同改；`train_scripts/v11/README.md:50` 改为「既有路径未改动，未提交状态见 §7.4」。事实表在 `:434-439`（五文件 `M`、`models/data/` 不入库、无 ckpt 快照、旧 run 不可考）。全文 `逐字节` 仅剩 **`:591`** 的 F3 对账行（引述旧措辞）。 |
| **F4** | 术语清零，统一「配置敏感度 0.007」 | **通过** | `README.md:39` 段「**配置敏感度 0.007（fp32↔fp16，确定性、可避免、不可平均）**：同一配置两次运行 1000/1000 逐位相同…fp32 与 fp16 是两条数值路径…**这不是随机噪声**」；`grep -rn "离散翻转分辨率\|噪声底" README.md Implementation.md train_scripts/v11/` 仅命中 `:455`（**作废声明**）与 `:592`（F4 对账行）。门槛句「任何 \|Δ\| ≤ 0.007 的均值差都不构成结论」已在 README 落地。 |
| **F5** | README 对外口径按 captain 定稿 | **通过** | `README.md:35`「**A0 20.0690 vs 20.084，落后 0.015（≈2× 配置敏感度）；本轮目标 +0.025 → 20.094 才算明确领先。** 对外统一写「落后 0.015」，不再写「打平」「领先」等混淆话术」；`README.md:28` 已换 **KaMIS 20.1020 @ 3.046 s** 并同时写两个可审计事实（`--time_limit=1` 只有 99/1000 图返回、那 99 图均值 19.8384；墙钟地板 1.87 s/图），旧「20.1060@1 s」明确标作废；`README.md:37` 20.0685 标 `historical_clock_reference_unreproducible`。`grep 打平 README.md` 仅剩 `:35` 的「不再写『打平』」与 v9 DCSS 行的「打平」（指验证目标，非对 RLSA 口径）。 |
| **负例** | 门 0 必须能失败（新 A0 下） | **通过（本地实跑）** | `python models/mis_v11_gates.py selftest` → **10/10**（含 `frozen_equivalence`）、`passed=true`、exit 0；`python models/mis_v11_gates.py gate0 --negative-selftest` → `clean_passes/shift_fails/config_fails` 全 true、exit 0。判死逻辑 `:564-566`：shift 必须 `checks['search'] is False`、config 必须 `dtype_fp16/dense/not_fp32_csr` 同 false（+0.05 位移与 fp16→fp32 两种破坏都真的会 fail，见 §6.4 的 monkeypatch 复验）。 |
| **fifo 等价** | selftest 第 10 项 ≥2 种子 × ≥1 合成图、失配非 0 | **通过** | `models/mis_v11_gates.py:180 frozen_equivalence(seeds=(1017,1019), graphs=3, rounds=6)`：3 合成图 × 2 种子，比对 `work/best/rule_size/exchanges/neutral_decisions/outer_steps` 六项逐位相同；结果 `frozen_equivalence=True` 进入 `checks`（`:263-269`），`passed=all(checks)` ⇒ 失配即 passed=false、exit≠0。 |
| **C4 措辞** | §14.10 追加更正、三点证据齐备 | **通过（上轮已完成）** | `RLNN_v11_Direction_Locked.md` §14.10.9（只追加）；Implementation `:215-216` 与 `README.md:46` 已收窄为「在同一移动族且起点已极大时，记忆只能重排/收缩候选、不能创造 +1 移动；M4 + 三重不动点 ⇒ 增益上界 0」；机制 ER=0.0668 / 命中 4493–63341、收益 gain +1.800→+1.745、起点 17.1 不可迁移三点齐备。 |

### 7.3 剩余问题（needs_revision 的最小集）

**F2-R（medium）§7.3/§7.4 正文仍把 27 张翻牌定因为「解码时刻」，与 §7.6 互斥**
- `docs/temporary/RLNN_v11_Implementation.md:401-402`：「真正成因是**解码时刻的 pad / 浮点累加路径差**」；
  `:450-452`：「**定因**：…差异不在引擎，而在**解码时刻**——最小驱动把 81 块一次喂完再跟一次解码；生产二进制按 `--decode_every 1000` 在末块之后解码」。
- 与 `:455-460`（术语段）和 §7.6 `:528-560`（「973/1000、27 张 ±1、净值 +8/1000 = **fp32 误配**造成」）冲突；
  也与 `:464` §7.5 自己的口径（27 张属于 fp32 误配跑）冲突。
- 为什么必须改：论文的机制句就是「配置敏感度 0.007 = fp32↔fp16 数值路径差」。同一份 Implementation 里若还留着
  「定因是解码时刻」，审稿人或后续 collector 会引用错那句；而 §7.4 的实验本身也不能推出「解码时刻是全部成因」——
  它只证明**引擎层无漂移**（HEAD vs 当前引擎在最小驱动下 18/27 相同）+ 生产二进制下 dtype 生效（0/27）。
- **修法（两处各一句，不动数字/表）**：
  1. `:401-402` 改：「但 §7.4 的三路对比**否掉了这个归因**：HEAD `6f3b624` 引擎与当前引擎在最小驱动下逐图一致（都 18/27 得旧值），说明**引擎层无语义漂移**；27 张的净差来自 **fp32 误配的数值路径（§7.6）**，最小驱动与生产二进制之间另有一层解码时刻的 ±1（见 §7.4）」。
  2. `:450-452` 的「定因」小标题改为「**驱动层对照（只回答『引擎有没有漂移』）**」，正文改为：「三路对比说明：引擎层无漂移；最小驱动 18/27 得旧值、生产二进制 0/27，差额属**驱动/解码路径**的 ±1。**27 张相对旧基线的净差（+8/1000）归因为 fp32 误配**（§7.6：声明配置 fp16 生产跑对同一旧基线是 999/1000 相同、仅 `graph_000466` +0.5）。」
  3. 表头 `:444`「27 张翻牌里得『旧值』」加脚注：「本表用最小驱动，**不改变** §7.6 的 dtype 定因；它只排除『引擎构建漂移』。」
- 不改判据、不需重跑：这是因果句归位，数字与门 0 结论全部不变。

**F3-R（low）`Implementation.md:591` 仍留「逐字节」字样**
- 位置是 §9 的 F3 对账行（引述旧措辞 + 修法），不是现行主张。修法：加引号写成「（旧措辞，已删）」，
  或改为「逐字/逐字段」以彻底消词。

### 7.4 复评实测命令（可复现）

```bash
cd F:/Code_Repositories_2/RLNN_lab
python models/mis_v11_gates.py selftest                    # 10/10, passed=true, exit 0
python models/mis_v11_gates.py gate0 --negative-selftest   # clean/shift/config 全 true, exit 0
grep -n "REFERENCE_SEARCH\|CONFIG_SENSITIVITY\|noise_floor" models/mis_v11_gates.py
grep -rn "逐字节" docs/temporary/RLNN_v11_Implementation.md train_scripts/v11/
grep -rn "离散翻转分辨率\|噪声底" README.md docs/temporary/RLNN_v11_Implementation.md
sed -n '401,402p;452,460p' docs/temporary/RLNN_v11_Implementation.md   # 冲突句现场
```

### 7.5 放行条件

F2-R 两处因果句归位（+ 可选 F3-R 消词）后即可 **pass → 解锁 t7**；F1/F3/F4/F5/负例/fifo 等价/C4 措辞
本轮已全部通过，复评时**只需再读那两段**，不必重跑任何东西（判据常量已冻结在 `:75-77`）。

### 7.6 复审后更正（captain 2026-11-17 23:43，覆盖 §7.3 的修法文本）

captain 纠了我在 F2-R 修法里提出的**三层结构**（「引擎层无漂移 + 净差归 dtype + 驱动层另有一层 ±1」）：
**第三层没有证据，不得写进文档，也不得写进论文。** 盘上事实是——声明配置（fp16+dense）对旧 20.0685 是
**999/1000 逐图相同**，唯一差异 `graph_000466` 可由步数差（775→810）解释；若「驱动层独立贡献 ±1」存在，
它在同一 dtype 下本应出现，但没有出现。因此完整解释就是 **fp32↔fp16 数值路径（配置敏感度 0.007）**。
§7.4 的三路实验只能推出「引擎层无漂移 + dtype 在产物里生效」，**推不出**解码时刻是成因——
这一点我在 §7.3 已写对，但修法句子超出了证据，撤回。
t19 按 captain 指令执行：§7.4 标题「驱动层对照：只回答引擎有没有漂移」、正文明说「推不出解码时刻是成因」、
净差一句归 fp32 误配并引 999/1000 + `graph_000466`；§7.3 的「解码时刻」降级为「驱动/解码配置差异之一，
未单独隔离」；`:591` 加「（旧措辞，已删）」。**数字与表格一律不动。**
复评口径不变：只读那两段 + 两条 `grep`，不重跑、不重开 GPU。

## 8. t13 复评（attempt 3，2026-11-17 23:51–）：t19 的因果句归位 — **PASS**

**复评范围（captain 限定）**：只读 §7.3/§7.4 两段 + 两条 `grep`（`逐字节`、`解码时刻`）；不重跑、不重开 GPU、不重开 R1–R7。
**对象**：`docs/temporary/RLNN_v11_Implementation.md`（mtime 23:50）。**未改任何代码或文档正文**（本节只追加）。

### 8.1 裁定

**pass。** §7 的 F2-R 与 F3-R 已按 captain 更严的写法（**不许引入「驱动层独立 ±1」这一未测层**）落地，数字与表格未动，t7 解锁。

### 8.2 逐条核验

| 项 | 要求 | 现状 | 结论 |
| --- | --- | --- | --- |
| F2-R-1 | §7.4 标题改「驱动层对照：只回答引擎有没有漂移」 | `:420`「### 7.4 驱动层对照：只回答「引擎有没有漂移」，**不定因**（HEAD 构建实测 + 仓库事实）」 | 通过 |
| F2-R-2 | 保留三路实验结果（18/27、18/27、0/27） | `:445-447` 表格完整：HEAD 引擎 18/27、当前引擎 18/27（「引擎构建层没有语义漂移」）、生产二进制 0/27 | 通过 |
| F2-R-3 | 正文明说「能推出引擎层无漂移 + dtype 产物生效；**推不出**解码时刻是成因」 | `:449-457`：「**能推出**「引擎构建层无语义漂移」与「dtype 确实在产物里生效」…**推不出**「解码时刻是全部成因」…这条差异与 fp32↔fp16 的数值路径差**叠在一起、本轮没有单独隔离**（隔离需固定 dtype、只改解码时机各跑一次 1000 图，未做）」 | 通过（且写明未做的隔离实验，未把「未隔离」写成「已排除」） |
| F2-R-4 | 净差一句归 fp32 误配，引 999/1000 + graph_000466 | `:459-463`「**净差归属（27 行翻牌）**：净差 **+8.0/1000 = 20.0765 − 20.0685** 由 **fp32 误配**解释——声明配置（fp16）与旧 20.0685 是 **999/1000 逐图相同**（唯一差异 `graph_000466` 19.5→20.0，由步数 775→810 解释…）；fp32 误配跑…**973 相同 / 18 行上升 / 9 张 −1 = +8.0 节点 = +0.0080**」 | 通过 |
| F2-R-5 | §7.3 的「解码时刻」降级为「驱动/解码配置差异之一，未单独隔离」 | `:400-404`：「成因只能记为**驱动/解码配置差异之一（未单独隔离）**：§7.4 证明的是「引擎层无漂移 + dtype 在产物里生效」，**推不出**「解码时刻就是全部成因」…净差归属见 §7.4 末段」；`:354` 早前段落同步为「不能把 ±1 单独归因到解码时刻」 | 通过 |
| F3-R | `:591`（现 `:602`）的「逐字节」加「（旧措辞，已删）」 | `:602`「| **F3** 「逐字节」（旧措辞，已删） | …」 | 通过（`grep -rn 逐字节` 仅此 1 处，且已标注作废） |
| 数字不动 | 20.0690 / 999 / 973 / +0.0080 / +0.0005 / +0.0020 / 810 步 | 逐项对照 §7 原值与 §7.5 台账一致；`:472` 只是把「27 张」细化为「27 行 = ±1 行 25 + 0.5 行 2」（16×(+1)+2×(+0.5)+9×(−1)=+8.0，与净值自洽，非改数） | 通过 |

`grep -rn "解码时刻"` 全仓现在仅 3 处，且**全部为否定/限定句**（`:354`「不能把 ±1 单独归因到解码时刻」、`:404`、`:454` 的「推不出…」），无一处把解码时刻当成因陈述。

### 8.3 复评执行记录

```bash
cd F:/Code_Repositories_2/RLNN_lab
ls -la docs/temporary/RLNN_v11_Implementation.md        # mtime Nov 17 23:50（t19 落地）
grep -rn "逐字节" docs/temporary/RLNN_v11_Implementation.md train_scripts/v11/   # 1 处，已标"（旧措辞，已删）"
grep -rn "解码时刻" docs/temporary/RLNN_v11_Implementation.md train_scripts/v11/ README.md  # 3 处，全为否定句
sed -n '396,412p;420,470p' docs/temporary/RLNN_v11_Implementation.md           # 两段逐句读
```

未重跑 `selftest`/`gate0`（判据常量与 gate 逻辑本任务未被触碰，`models/mis_v11_gates.py` mtime 仍 22:36；
上一次实跑 10/10 + 负例 3/3 的结论继续有效）。

### 8.4 结论与解锁

- t13 三次 attempt 的净结果：t11 堵住三处判据级假阳性（signal-only、恒真 dtype、final_size 判据）+ 负例可失败；
  t16 把 A0 从 fp32 误配的 20.0760 对齐到声明配置的 20.0690（验收带回到 [20.0590,20.0790]）；
  t19 把「27 行 ±1 的成因」归位为唯一的 fp32↔fp16 数值路径，且明确标注未做的隔离实验。
- **t7 解锁条件满足**：门 0 判据（A0 20.0690 ± 0.01、search_size only、810 步硬校验、dtype/dense/fp32_csr 读产物、
  负例可 fail）与「不留两个机制句」两项均已成立。
- 遗留（不阻塞）：F6 `summarize_v11.py:26` 仍经 `eval_v8_capture_budget` 拉 tqdm 链（低优先，GPU 机可跑）；
  §7.5 的「行/张」措辞已自洽但建议论文引用时统一写「图（graph）」，避免 repeats=2 下混淆。
