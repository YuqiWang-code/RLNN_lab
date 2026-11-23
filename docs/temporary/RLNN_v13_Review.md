# t23 复审——v13 融合引擎（审 t21）

裁决：**needs_revision**（2 medium + 3 low；无 blocker。加速本身站得住，问题在头条口径与门/文档一致性）
范围：只读产物 + CPU 重算（未开 GPU）。任务书里的 `models/mis_v14_fused.py` / `train_scripts/v11/v14_engine.sh` / `fused_bench.json` 均不存在；实际实现 = `models/eval_v8_padded.py` 四个 additive flag（`--reserve_ms` :137/:236-239、`--compile_model` :142/:194、`--skip_incumbent` :148/:274、`--no_pave` :154/:234/:303）＋ `models/probe_p1_fusion.py` / `models/probe_p05_budget.py` / `train_scripts/v11/v14_{fused,ab,ab01,l1l2}.sh`。

| # | 检查项 | 结果 | 证据 |
|---|---|---|---|
| 1 | 冻结文件 | 通过（附条件） | `models/mis_v3_search.py`、`models/mis_v3_value.py` 未改；五个未提交文件是 v9–v11 遗留 + v13 接线混在一起。四 flag 全关时 `decode` 落回 `kind=="eager"` 原分支（`eval_v8_padded.py:253-261`，`recombine_top` 分支在前、取 0 时不触发）、`reserve`(:234-239)、`pave_deadline`(:296-298)、`final`(:303-304) 与旧式等价。**v13 后未重跑默认路径逐位对照**（门 0 产物 05:58 早于接线），仅由 `numeric200_plain` vs t24 旧产物 200/200 相同间接担保。 |
| 2 | 数值等价 | 通过但贴线 | 我按 CSV 重算：194/200 相同、4 图 +1、2 图 −1、max\|Δ\|=1、\|Δ\|≥2 **0 张**、不同图 3%≤5%。未发现「先出好数再倒推门槛」：门槛写于 `RLNN_v13_Engine_Rewrite_Spec.md` §1（05:12），门跑于 06:45。但 **Δmean = 19.9250(plain) → 19.9150(compiled) = −0.010，正好压在 ±0.01 边界，全文未报有符号均值差**。 |
| 3 | 提速归因 | 通过 | 省 303 µs ≈ pointwise 425 µs（t17 `e3_profile/kernels.json`：94.9 kernel/步、pointwise 425/24 kernels、gemm 345、launch gap 仅 43 µs）；纯 launch 优化最多值 43 µs。步数未缩（gate 两臂均 130；0.2 s 232.6→366.9；1 s 1851.8）；decode 节奏 6 臂一致（`--decode_every 1000 --decode_top 4`）；计时点未挪（L3 0.531×1851.8=983.3 ms ≈ wall 983.7）。 |
| 4 | 可比性 | **不通过**（R1） | 1 s：20.0870 vs RLSA 20.084（+0.003，判持平）、vs KaMIS −0.015，§7 明写不得写「领先」，✓。0.2 s：B+ 20.0327 vs RLSA 19.9620@0.201 s（+0.0707，同墙钟）✓，但 **B+ 用 `--no_pave`**，违 spec §2.1「SOTA 主口径必须含 load+decode+PAVE」；PAVE-on 的 0.2 s 只有未融合的 A=19.9560（−0.006）。 |
| 5 | 可复现 | 通过（附条件） | `padded_ref*.json` 记 `leading_decode/pave/fixed_steps/compile_model/reserve_ms`（`eval_v8_padded.py:359-370`），v11 CSV 还逐行记 `dtype/dense/fp32_csr/skip_incumbent/no_pave/compile_model`（`:429-430`）；但 summary 里**缺 dtype/dense**、缺 inductor/compile 版本。`v14_fused.sh` 写死服务器绝对路径（服务器脚本可接受）。B 臂 CSV 少 `compile_model` 列，合并均值须按列名容错。 |
| 6 | 结论↔产物 | **一处不一致**（R3） | L1/L2 逐图 1000/1000 相同 ✓、L3 20.0900/1851.8/0.531/wall 983.7 ✓、B+/B ✓、`drift_vs_B` 2823/3000、≥2 占 0.033%（1 图）、max 2 ✓；但 §9 交接写「\|Δ\|≥2 只有 1 张」，与 §6 及产物（0 张）矛盾。 |

## Findings

- **R1（medium，blocker-for-headline）** `docs/temporary/RLNN_v13_FusedEngine.md:212`/`:218`/`:293`/`:339`：0.2 s 头条 20.0327 是 PAVE-off 口径（`--no_pave`），不符合 spec §2.1「主口径必须含 load+decode+PAVE」；融合+PAVE-on 的 0.2 s 从未实测。修法二选一：(a) 补一格 `0.2 s × fusion × PAVE-on × 3 seed`（≈0.3 GPU·h，预估 ≈20.00，仍 +0.04）；(b) 头条改标 `search-only`，与 A 臂分表，正文不得与 RLSA 的含 PAVE 口径并列。
- **R2（medium）** `RLNN_v13_FusedEngine.md:199` 与 `:36`/`:303-304`：`:199` 原句「G1（≥1.4×、**≤0.59 ms**）在 200 图端到端口径下**也过**：0.910 → 0.637」本身自相矛盾（0.637 > 0.59）；§7 随后把 G1 改成「倍率即门」（`:303-304`）、称 0.59 是「0.867/1.4 的粗算代理，已作废」，但 `0.867/1.4=0.619≠0.59`，且 t24 的原文（`:36`）仍留在同一文件里。时序：改门文字晚于 `p1_fusion.json`（06:42），两处口径并存 ⇒ 读者会看到「按绝对值不过、按倍率过」。修法：两口径并列写结论差异（比值 1.43× 过；绝对 ≤0.59/≤0.619 不过，因基线臂不同 0.910 vs 0.867 也非同一度量），删掉 `:36` 的旧门槛或标注已被取代，并由 captain 在文档里签认「G1 采用倍率口径」。
- **R3（low-medium）** `RLNN_v13_FusedEngine.md:331`（§9 交接）：`|Δ|≥2` 写「只有 1 张」，把 `drift_vs_B`（B vs B+，3000 对里 1 图 Δ=2）与 `numeric200`（plain vs compiled，200 图里 **0** 图）两本账混在一起；§6 `:205` 自己写的是「\|Δ\|≥2 0、max 1」。§9 是下游引用处，先改这里。
- **R4（low）** `RLNN_v13_FusedEngine.md:204`：只写「194/200 相同」，未报 Δmean=−0.010（贴 spec §1 上限）；同表「与基线同解 是」掩盖了 `p1_fusion.json` 里 `compiled_model/compiled_body` 的 `energy_matches=false`（−17.92 vs −16.97）——同尺寸但轨迹不同，须写「尺寸同解、能量轨迹不同」。
- **R5（low）** `models/eval_v8_padded.py:359-370`（`padded_ref*.json` 的 summary）：记了 `leading_decode/pave/fixed_steps/compile_model/reserve_ms`，但**不记 dtype/dense**（只在 v11 CSV 的 23 列里），也不记 inductor/compile 版本；另：融合后**没有 kernel 计数**（`probe_p1_fusion.py` 只报 ms/步），故论文计划里的「kernel 95→≤10」目前无产物支撑。修法：summary 加 `dtype/dense` 两键；用 `torch.profiler` 在同一探针对 compiled 版数一次 kernel/步，写进 `p1_fusion.json`。

## 正向结论（可直接用）
- A0 默认路径行为保持（四 flag 全关等价，`reserve`/`pave_deadline`/`final` 三处逐行核过），A0 台账 20.0690 未被重写污染。
- 1 s 的「饱和」结论与产物完全一致（L2−L1 逐图 1000/1000 相同，PAVE 增益精确 0；L3−融合臂 +0.0010/Δsteps +127.6），且 §7 主动不写「领先 RLSA」，口径自觉。
- 加速是真实 kernel 消除（303 µs ≈ 71% pointwise），不是缩步/挪计时/少 reduce；`deadline_overrun` L3 仅 1 图 >0。
