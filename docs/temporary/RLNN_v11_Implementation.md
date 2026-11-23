# RLNN v11 实现与闸门报告（t5）

> 依据：`docs/temporary/RLNN_v11_Direction_Locked.md`（v3 定稿，§14 为唯一生效补充）+ `docs/temporary/RLNN_v11_MATLAB_Validation.md`（t4 五条输入）+ captain 三条技术校正（K 关线固定 200、门 0 判据按实测校正、`unroll_steps` 200 带自适应门）。
> 机器：远程训练机 GPU1 = RTX 5090 32 GB（`CUDA_VISIBLE_DEVICES=1`，单进程）；`torch 2.9.1+cu130`；python `/home/hzeng/envs/rlnn/bin/python`；仓库 `/home/hzeng/project/ZH/RLNN_lab`。
> 状态：**engine 闸门 7/7、CPU 自检 10/10（含 `frozen_equivalence`）、部署冒烟与 `--fixed_steps` 透传全绿；梯度自证 20 图名义过、200 图判死 ⇒ C3 判死；C4/L04 动作级 FIFO 已按 captain 复核修好键语义与 gain（`tabu_walk_keyfix/`）：掩码确实生效（ER=0.0668 在窗内、FIFO 命中 4493/63341），但 gain 反而 −0.055、refresh 不增 ⇒ 机制成立、收益不成立（§3.6，结案措辞见该节末）；门 0 t11 定稿：声明配置（fp16+dense、固定 810 步、1000 图）**8/8 全绿 pass**，A0 实测 search **20.0690** / final **20.0705**，与对照臂 999/1000 图逐位相同（§7.6）；先前「+0.008 / 27 行翻牌」是脚本漏传 `--mixed_precision fp16` 跑成 fp32 造成的配置偏差，被 R2 的数据驱动 dtype 检查当场判死（§7.6）。**

---

## 0. 一句话结论

1. 新代码全部落在新文件里（`models/mis_v11_*.py`、`models/probe_*.py`、`train_scripts/v11/*`），冻结协议文件的改动是 **5 处 additive**，每条既有代码路径未改动（既有代码的**未提交状态**见 §7.4——`git diff` 不能证明零改动；门 0 用同一份代码复现基线来证明行为等价）。
2. **引擎层没有漂移**：captured 引擎与参考实现逐位一致（`eager` vs CUDA-graph、`unroll` vs 引擎单步、解码 size 与 A.3 参照一致、deadline 语义、固定步数、同种子可复现），7 项全绿。
3. **C3 的「可微 size-aware 目标」在数学上不成立**：解码大小对参数是分段常数，任何 straight-through 梯度都是穿台阶的幽灵；实测 −∇/+∇/同范数随机扰动的 Δsize 几乎相同（200 图：+0.020/+0.0165/+0.014，Spearman 0.168）⇒ 梯度对解码大小**没有方向信息**，§2.9 门 1 判死，A1/A1′ 不得开。
4. **门 0 在声明配置下 8/8 通过，且顺带抓出一个配置缺陷**：`--fixed_steps 808`（810 步、1000 图、fp16+dense）→ search **20.0690** / final **20.0705**，`gates_gate0.json` 八项全绿、exit 0；同命令两次独立运行逐图 1000/1000 相同。与三周前对照臂比 **999/1000 图逐位相同**（仅 `graph_000466` +0.5）。先前报的「973/1000、27 行翻牌、均值 +0.008」是**脚本漏传 `--mixed_precision fp16`**（跑成 fp32+dense，search 20.0760）造成的数值路径差，被 t11 的 R2 检查（从 `--v11_csv` 行读 dtype）当场判 `dtype_fp16=false`、exit 3，修好后重跑即全绿（§7.6）。⇒ 术语定为 **配置敏感度 0.007（fp32↔fp16）**：同一配置的运行间抖动是 0（1000/1000 逐位相同），fp32 与 fp16 之间的 0.007 是一次**确定性、可避免**的配置差异，既不能平均、也不该塞进容差。门 0 判据 `|Δ search_size| ≤ 0.01`，**A0 参考 = 20.0690（声明配置，t16 定稿）**；20.0760 是 fp32 误配，只作配置敏感度证据（t13-F1 已把它从 `REFERENCE_SEARCH` 撤掉：偏 +0.007 叠 ±0.01 会把 20.080–20.086 的真实退化放过去）。
5. **C4/L04 的动作级 FIFO：机制成立、收益不成立。** 第一版把 FIFO 键写成「完全相同 removed 集合」，键空间大到永不命中（ER≡0，六档 h 逐位相同）——这是实现缺陷，已按 captain 复核改为**节点身份**的交叉集 tabu（L04 Alg.1）。修正后掩码确实 fire（FIFO 命中 4493/63341 = 7.1%，ER=0.0668 落在 [0.05,0.6]，形式判据通过），但 best 从 18.9050 掉到 18.8500（gain +1.800 → +1.745）、`refreshed_rounds` 0.095 → 0.090 ⇒ 记忆不是瓶颈，「走动 → +1」的通道没被打开，h=8…128 逐位相同说明深窗口无信息（§3.6）。

---

## 1. 改动清单（文件:行 → 新增了什么）

### 1.1 新文件（无冻结风险）

| 文件 | 行数 | 内容 |
| --- | --- | --- |
| `models/mis_v11_unroll.py` | 325 | C3 展开器 + 训练钩子 + `run.json`。`DEFAULT_STEPS=200`(:36)、`FALLBACK_STEPS=128`(:37)、`DECODE_EVERY=10`(:38)、`MEMORY_GATE_BYTES=6 GiB`(:39)、`TIME_GATE_SECONDS=0.6`(:40)；`_rand`(:46)、`_dense`(:52)、`decode_size`(:73)、`unroll`(:80)、`unroll_objective`(:156)、`_rollout_seed`(:186)、`generator_for`(:192)、`configure`(:201)、`_cfg`(:211)、`unroll_graph_loss`(:227)、`_jsonable`(:279)、`dump_run_json`(:291)、`config_snapshot`(:318) |
| `models/probe_unroll_grad.py` | 201 | §2.9 梯度五项自证（20/200 图）。`probe_parser`(:40)、`spearman`(:54)、`feasible`(:64)、`run_probe`(:70)、`main`(:187) |
| `models/mis_v11_gates.py` | 453 | 四模式硬闸门 `selftest\|engine\|gate0\|audit`。常量 `REFERENCE_SOLUTION=20.0685`(:46)、`REFERENCE_TOLERANCE=0.0005`(:47)、`STEP_RANGE=(700,860)`(:48)、`FIXED_STEPS=808`(:49)、`FP32_CSR_MS=3.9`(:50)、`FP16_DENSE_MS=(1.05,1.30)`(:51)；`unpack`(:57)、`independent`(:61)、`reference_problem`(:68)、`build_cpu_graph`(:114)、`selftest`(:137)、`engine_gates`(:205)、`gate0`(:342)、`audit`(:381)、`main`(:401) |
| `models/mis_v11_tabu.py` | 408 | C4/L04 **节点身份**动作级 FIFO（不碰 `mis_v3_search.py`）。`_Entry`(记录 added/removed 两元组)、`TabuList`(h 深度 FIFO，`cross`=L04 Alg.1 交叉集禁止 / `same`=同集读法，含 `checks/hits/sample` 计数与命中样例)、`improving_actions_tabu`（**先过滤后截断**，否则 ER 无意义）、`repair_tabu`（补上冻结版 `:247 repair()` 缺失的 tabu 参数并 push 已执行动作）、`pool_plain_and_tabu`（配对池 + `rejected` 计数，ER 定义在此）、`platform_search_tabu`（跨轮 FIFO + 逐轮 `on_round` 回调算 ER/AR）、`rollout_tabu`、`self_test()`（A：undo 必被 cross 禁、same 不禁；B：真实候选池上构造必然触发的例子，断言 `pool_tabu ⊊ pool`） |
| `models/probe_tabu_walk.py` | 234 | C4 探针：`--h` 多深度 × `--tabu_mode cross,same` × `--graphs` × `--outer_rounds`，输出 `tabu_walk.json/.csv`（JSON `formulas` 字段写死 ER/AR 定义），可行性断言 `feasible()`；每图记录真实 `gain`、`refreshed_rounds`、`er_rejected`、`er_planned/er_pooled`、FIFO `checks/hits`；判据是合取（AR≥1/8 且 ER∈[0.05,0.6]），失败按 `kill_reason` 区分 `tabu_never_fired` / `ar_zero` / `no_refresh` / `er_out_of_window` 并 `SystemExit(3)` |
| `models/summarize_v11.py` | 96 | 接现有配对统计：读两份 `--v11_csv` 逐图 CSV，按 (graph_id,budget) 配对 + 三 seed pooled 5000 次自助（复用 `eval_v8_capture_budget.paired_bootstrap`），直接按 §2.7/S1 判 `pass/signal-only/unstable/fail`（含「任一 seed < −0.010 判不稳」） |
| `train_scripts/v11/probe_grad.sh` | 28 | 梯度自证（20 图/K=20/40 步/seed 2027） |
| `train_scripts/v11/a0_gate0.sh` | 71 | 门 0 双口径（(i) `--fixed_steps 808` 主闸门；(ii) 时钟口径只报告，退出码取 (i)） |
| `train_scripts/v11/c3.sh` | 77 | 训练+评测启动器：`bash c3.sh <gpu> [unroll\|erdoes] [train\|eval\|all]` |
| `train_scripts/v11/smoke4.sh` | 63 | 4 图冒烟：engine 闸门 → 部署口径 → `--fixed_steps 810` 精确性 |
| `train_scripts/v11/README.md` | 40 | 执行顺序、门 0 双口径说明、GPU 纪律 |

### 1.2 冻结协议文件的 additive 改动（5 处，逐条列）

| # | 文件:行 | 新增了什么 | 既有路径是否变化 |
| --- | --- | --- | --- |
| 1 | `models/solvers.py:21` | `from mis_v11_unroll import unroll_graph_loss  # v11 C3: additive, used only when --loss unroll` | 无（纯 import） |
| 2 | `models/solvers.py:405` | `elif self.loss == 'unroll': loss = unroll_graph_loss(self, graph, A, normalized, x)`，插在 `elif self.loss == 'reinforce'` 之后、`else: raise NotImplementedError` 之前 | 无（`erdoes`/`reinforce` 分支一字未动） |
| 3 | `models/main.py:38` | `--loss` choices 由 `['erdoes','reinforce']` 扩为 `['erdoes','reinforce','unroll']` | 无（默认仍 `erdoes`） |
| 4 | `models/main.py:39-42` | 新参数 `--unroll_steps`(200)、`--unroll_decode_every`(10)、`--unroll_archive`、`--run_json` | 无（默认不触发） |
| 5 | `models/main.py:104-110` | `if args.loss == 'unroll' or args.run_json: configure_v11(solver, args)`（在 `solver.train()` 之前）+ `if args.run_json: dump_run_json(...)`（在 `train()` 之后） | 无（`loss != unroll and not run_json` 时整段跳过） |

`models/models.py`、`models/mis_v1.py`、`models/utils.py`、`models/problems/`、`models/mis_v3_search.py`、`models/mis_v3_value.py`、`models/eval_v1/v2/v3.py` **一个字节未改**；`models/mis_v8_pad.py` 未改（C4 走新模块，不碰部署引擎）。

> 工作树提醒：仓库在 t5 之前**已经带着前序任务的工作树改动**（`models/models.py`、`models/mis_v8_pad.py`、`models/solvers.py` 的 `--arch scale/norm` 分支、`models/eval_v8_padded.py` 的 `--lns_*` 与 `decode_top_*` 等），`git diff --stat` 看到的大数字不全是 t5 引入的。t5 在冻结文件里新增的 hunk **只有上表 5 处**，在 `models/eval_v8_padded.py` 里新增的只有 `--fixed_steps`/`--v11_csv`/`pave_deadline`/v11 CSV 写盘四段（`import math` 供 `ceil`；`import numpy as np` 供前序 `decode_top_*` 的 `np.packbits`）。

### 1.3 非冻结的评测框架改动（`models/eval_v8_padded.py`，1 处参数 + 1 处 CSV）

| 文件:行 | 新增 | 说明 |
| --- | --- | --- |
| `:13` | `import math` | `ceil` 用 |
| `:137-139` | `--fixed_steps`（int，0=deadline 驱动） | §2.10 的 3 行透传 |
| `:140-142` | `--v11_csv`（另写逐图 CSV） | v11 列：`graph_id,budget,cell,arm,decoder,seed,repeat,search_size,final_size,steps,wall_ms,ms_per_step,search_ms,load_ms,blocks,pave_gain,deadline_overrun,temperature,anneal,fixed_steps,dtype,dense,fp32_csr`（23 列） |
| `:211-218` | `if args.fixed_steps: cutoff=inf; fixed_blocks=ceil(fixed_steps/chunk)` | `--fixed_steps 0` 时 `fixed_blocks=None`，`cutoff` 与既有完全一致 |
| `:264-271` | `pave_deadline = deadline if fixed_blocks is None else perf_counter()+reserve`，`run_pave(..., pave_deadline)` | 步数口径下 PAVE 仍拿墙钟预留；时钟口径下 `pave_deadline is deadline`，行为等价（既有路径未改动，未提交状态见 §7.4） |
| `:377-393` | `if args.v11_csv: DictWriter(extrasaction="ignore")` 另写 | **既有 CSV/JSON/summary 完全不动**（A0 可比性的前提） |
| `:377-400` | v11 CSV 再增 `dtype`(`solver.dtype`)/`dense`(`engine.use_dense`)/`fp32_csr`(派生) | t11 R2：门 0 读**数据**不读 flag。既有 CSV 仅「新增 5 列」（`fixed_steps,dtype,dense,fp32_csr` + 上表末列），按列名读的消费者不受影响 |

---

## 2. 运行方法

```bash
# 本地（Windows）：语法与静态检查
python -m py_compile models/mis_v11_unroll.py models/mis_v11_gates.py models/probe_unroll_grad.py \
                      models/mis_v11_tabu.py models/probe_tabu_walk.py models/main.py models/solvers.py
python models/mis_v11_gates.py selftest                 # CPU，9 项，不需要 GPU/数据
python tools/server.py push                             # git ls-files --cached --others --exclude-standard

# 服务器 GPU1（一律单进程、不碰 GPU0）
bash train_scripts/v11/smoke4.sh 1                      # engine 闸门 + 部署口径 + fixed_steps 精确性
bash train_scripts/v11/probe_grad.sh 1                  # 梯度五项自证（20 图）
bash train_scripts/v11/a0_gate0.sh 1                    # 门 0 双口径（1000 图 × 两条）
bash train_scripts/v11/c3.sh 1 unroll train             # A1 训练（loss=unroll，200 步、每 10 步解码）
bash train_scripts/v11/c3.sh 1 unroll eval              # A1 终评（1000 图 × 1.0/0.2 s × seed 1017/1019/1021）
bash train_scripts/v11/c3.sh 1 erdoes train             # A1′ 对照（只差 --loss，其余 argv 由 audit 模式校验）

# 配对统计 + S1 判据（读 --v11_csv 的逐图 CSV）
python models/summarize_v11.py --a outputs/v11/A0/v11_rows.csv --b outputs/v11/A1/v11_rows.csv --boot 5000

# C4/L04 探针（纯 CPU，不占 GPU）
python -u models/probe_tabu_walk.py --graphs 200 --outer_rounds 8 --h 8,16,32,64,128 \
  --budget 0 --result_dir outputs/v11/tabu_walk_200
```

训练协议的硬约束（**读码得来，违反会直接 raise**）：

- `models/solvers.py:333` `if self.graph_batch_size > 1 and self.loss != "erdoes": raise ValueError(...)` ⇒ `--loss unroll` 只能 `--graph_batch_size 1`；
- `models/solvers.py:337` `if args.do_train and (self.freeze_backbone_epochs or self.finetune_lr or self.validation_seed is not None): if self.graph_batch_size < 2: raise ValueError("V2 training requires multiple-graph MIS erdoes updates")` ⇒ unroll 训练**不能**同时传 `--validation_seed/--finetune_lr/--freeze_backbone_epochs`。

---

## 3. 闸门结果

### 3.1 CPU 自检 9/9（`python models/mis_v11_gates.py selftest`）

`finite` / `nonzero_grad` / `feasible` / `decode_matches_reference`（与 `problems.mis.Problem.decode_gradient_flip` 逐位一致）/ `decode_from_feasible_stays_feasible` / `archive_reported` / `same_seed_bitwise`（同种子两次 rollout 逐位）/ `degrade_recorded`（`run.json` 记 `degraded_reason`）/ `degrade_trace`（把 `TIME_GATE_SECONDS` 临时设 −1 强迫降级：`measured[0].steps==200` → 之后 `steps==128`、`degraded_reason` 非空、`updates==2`）。

### 3.2 engine 闸门 7/7（4 图，GPU1，`outputs/v11/smoke/gates_engine.json`）

| 检查 | 内容 | 结果 |
| --- | --- | --- |
| `eager_vs_captured_bitwise` | `eager_step(40)` vs captured `run(inf, max_blocks=4)`：state/best/energy 逐位 | pass |
| `unroll_vs_engine_step_bitwise` | 同一噪声流下 `mis_v11_unroll.unroll(steps=1)` vs `engine.eager_step(1)` 的 state 逐位 | pass |
| `decode_size_matches` | captured `decode_best` 的 `bit_count` == 参照 A.3 `decoded.sum(0).max()` | pass |
| `engine_decode_feasible` | 解码结果满足独立集断言 | pass |
| `deadline_ok` | `budget=0.05 s`：`wall ≤ budget + 1.5×block_time` 且 `steps>0` | pass |
| `fixed_steps_ok` | `max_blocks=ceil(808/10)` ⇒ `steps == blocks×chunk` | pass |
| `same_seed_reproducible` | 同 seed 两次 run 的 mask 列表相等 | pass |

修掉的 3 个自身 bug（记录在此，避免后来者重复踩）：`policy_parser().parse_args([])` 缺 `--result_dir`；报告行引用未定义的 `fixed_steps`；**`eager_step` 跑完会恢复缓冲，步进后的 state 只能取返回值**，不能读 `engine.state`（此前 425/4320 位差异全部来自读错对象，logits 逐位 diff=0.0 已证）。

### 3.3 部署口径 4 图冒烟（`train_scripts/v11/smoke4.sh 1`）

| 臂 | 解 | 步数 | ms/步 | load | decode |
| --- | --- | --- | --- | --- | --- |
| padded/captured 1.0 s | 19.25 | 875 | 1.082 | 5.72 ms | 1.15 ms |
| padded/cached 1.0 s | 19.25 | 860 | — | — | — |
| padded/captured 0.2 s | 19.00 | 105 | — | — | — |
| `--fixed_steps 810` | — | 810（断言 `steps==[810]`） | — | — | — |

`outputs/v11/smoke/timing/v11_rows.csv`、`outputs/v11/smoke/fixed/v11_rows.csv` 均按要求字段写出；既有 `padded_ref0.csv` / `summary json` 未被触碰。

### 3.4 梯度自证（§2.9）：20 图「表面过」、200 图判死

修了一个自己造的 bug：base 与 ±∇ 重放原先**共用同一个 generator 对象**（噪声流已推进），Δ 主要是噪声差。改成每次重放都用 `generator_for(device, probe_seed+index)` 重建，并新增 `delta_control`（同范数随机扰动）与 `determinism`（同 seed 同扰动必须逐位）。

| 图数 | Δ(−∇) | Δ(+∇) | Δ(随机对照) | grad/control | Spearman(‖g‖,Δ) | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | +0.0025 | −0.0125 | −0.0375 | 1.067 | 0.224 | 名义 pass（但 20 图自助 CI 全跨 0：Δ− 95%CI [−0.060,+0.070]，逐图 sd≈0.15=3 个节点） |
| **200** | **+0.0200** | **+0.0165** | **+0.0140** | **0.429** | **0.168** | **fail**（`direction=false`、`correlation=false`、`passed=false`） |

机制：解码大小是参数的分段常数函数，几乎处处梯度 0；straight-through/argmax 给出的非零梯度是「穿台阶的幽灵」，方向任意。200 图里 −∇、+∇、同范数随机三者 Δ 几乎相同（+0.020/+0.0165/+0.014），即扰动幅度而非梯度方向决定结果。`scale` 项也没过（‖g‖/‖logits‖ = 9.1e−5–1.4e−4，5/20 图低于 1e−4 下界；§2.9 只要求 1–4 项，故 20 图仍记 pass）。

### 3.5 C3 训练路径：先崩、再 NaN、最后有界但「只剩单步信用」

| 阶段 | 现象 | 修法 |
| --- | --- | --- |
| 首跑 | `AttributeError: 'RLNN' object has no attribute 'args'` | `MetaSolver.__init__` 不保存 namespace ⇒ 新增 `configure(solver, args)`，由 `main.py:104-106` 在 `train()` 前调用 |
| 二跑 | tqdm 87.97 s/it，`loss: nan`；`run.json` 已正确降级（200→128，`degraded_reason='per_graph 0.887s>0.6s'`，显存 4.75 GB 未触发显存门） | — |
| 定位 | 前向 loss 有限（−16.95/−17.50），**首次 `backward()` 的参数梯度即 nan** | 逐条 detach 三条复合路径 |
| 爆炸 ① | 状态载体的 `where` 分支对 x 的雅可比是 ±1（不衰减） | `flip = where(rr<prob, 1-x.detach(), x.detach())` |
| 爆炸 ② | policy 输入再吃一次模型雅可比 | `logits = flip_logits(x.detach(), ...)` |
| 爆炸 ③ | 解码器是到不动点的循环，每轮乘一次邻接矩阵（RB 平均度 ~80） | 解码一律 `x.detach()`，只作报告 |
| 结果 | 最大 \|param grad\|：所有路径未修时 128 步 NaN；修①后 1.9e14@40、NaN@128；修②后 7.1e12@40；修③后 **24.3@10 / 15.6@40 / 13.5@128（与展开长度无关）** | 训练冒烟 `loss −17.175`、`valid_obj −20.0`（此前 nan）；阈值门照旧工作（0.688 s>0.6 s ⇒ 降到 128、显存 54 MB） |

结论（已写进 `models/mis_v11_unroll.py:80-130` 与 `:156-185` 的注释）：**size-aware 目标在本代码库只能拿到「单步信用」**，真正的 BPTT 数值不可行；这不是实现 bug，而是 C3 的死亡证明。

### 3.6 C4/L04 动作级 FIFO 探针（CPU，200 图 × 8 轮，两轮：`tabu_walk_200/` 键错版、`tabu_walk_keyfix/` 修正版）

**第一轮（`outputs/v11/tabu_walk_200/`）的 ER≡0 是实现缺陷，不是物理结论。** FIFO 键写成
`tuple(action.removed)`、过滤条件写成「removed 集合与历史**完全相同**才禁」，而交换动作的 removed
集合几乎不重复，于是 h=0/8/16/32/64/128 六行逐位相同、ER 恒 0。把它当「ER 出窗」判死是错的 —— 这一条已
按 captain 的代码复核（m01061）撤回。

**修正版（`models/mis_v11_tabu.py`，`TabuList`）**：FIFO 记录 `(added, removed)` 的**节点身份**，
过滤用 L04 Alg.1 的交叉集：

* `cross`（默认）：`set(action.added) & recently_removed` 或 `set(action.removed) & recently_added`
  非空即禁 —— 也就是「刚被逐出的节点 h 步内不许重加，刚加入的节点 h 步内不许逐出」，undo 被禁；
* `same`：`added & recently_added` 或 `removed & recently_removed` 非空即禁（复核意见的字面读法）。

同时补上三个此前缺失的量：`fifo_checks/fifo_hits`（FIFO 实际拒绝次数）、`refreshed_rounds`
（刷新 archive-best 的轮数）、真实 `gain`（旧版把 `gain` 硬编码成 0）。自检
`python models/mis_v11_tabu.py` → `[A] undo forbidden: cross=True same=False`、
`[B] |pool|=24 |pool_tabu|=23 rejected=1 strict=True`（构造必然触发的例子，断言 `pool_tabu ⊊ pool`）。

公式（同时写进 `models/mis_v11_tabu.py` 与 `models/probe_tabu_walk.py` 的 docstring、以及结果 JSON
的 `formulas` 字段）：

```
ER = 1 - |pool_tabu| / |pool_plain|        配对于同一 (graph, state, outer round)
     分母 = 真正建过候选池的 macro 轮数（er_rounds），每轮各贡献一个比值；
     另报 er_pooled = 1 - Σ|pool_tabu| / Σ|pool_plain|，防止单个大池主导均值。
AR = |{轮次: size_after == size_before 且 mask 改变 且 未刷新 archive}| / outer_rounds
```

实测（`outputs/v11/tabu_walk_keyfix/tabu_walk.json`，200 图 × 8 轮，budget=0 确定性，wall 226 s）：

| mode | h | best | gain | AR | refresh 轮数/图 | ER | ER pooled | FIFO 命中/检查 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cross | 0（对照） | 18.9050 | +1.8000 | 0.9881 | 0.095 | 0.0000 | 0.0000 | 0 / 0 |
| cross | 8 | 18.8500 | +1.7450 | 0.9888 | 0.090 | **0.0668** | 0.0663 | 4493 / 63341 |
| cross | 16/32/64/128 | 18.8500 | +1.7450 | 0.9888 | 0.090 | 0.0668 | 0.0663 | 4493 / 63341 |
| same | 0（对照） | 18.9050 | +1.8000 | 0.9881 | 0.095 | 0.0000 | 0.0000 | 0 / 0 |
| same | 8..128 | 18.9050 | +1.8000 | 0.9881 | 0.095 | 0.0046 | 0.0045 | 138 / 61517 |

`verdict = {best_ar: 0.98875, best_refreshed_rounds: 0.095, best_fifo_hits: 4493, ar_threshold: 0.125,
er_window: [0.05, 0.6], er_ok: true, passed: true, kill_reason: null}` —— **形式判据（AR≥1/8 且
ER∈[0.05,0.6]）通过**。

但实质结论要分两层写清：

1. **掩码确实生效**：cross 模式下 FIFO 拒绝 4493/63341 = 7.1% 的候选检查，候选池缩小 6.7%（ER=0.0668
   落在窗口内），且出现了具体的节点级命中（JSON `fifo_sample`：`action_removed=[132]` 与
   `tabu_added=[6,132,164]` 相交，即「刚加入的节点 132 立刻被逐出」这条规则）。
2. **但增益与刷新都没改善**：cross 的 best 从 18.9050 掉到 18.8500（gain +1.800 → +1.745，**−0.055**），
   `refreshed_rounds` 0.095 → 0.090（略降），AR 0.9881 → 0.9888（同量级）。也就是说：**记忆机制是活的，
   但它没有打开「走动 → +1」的通道**，反而因为禁掉了部分候选池使个别图少拿 1 个节点。h=8…128 逐位相同
   说明这 8 轮里 FIFO 只覆盖了最近 ≤8 个动作，更深的窗口根本用不上。

判读建议（留给 captain 裁决，§3.5 的次级判据正对这种情况：AR>0 但 SIZE 不变 ⇒「候选池非空但无增益」）：
C4 的**机制假设成立**（FIFO 会 fire），**收益假设不成立**（gain 不增、refresh 不增）。因此不建议为它花
§3.6 的 0.6 GPU·h 上 1 s 臂；若仍要上线，唯一可测的形态是「只禁 undo 的 h=1」而不是 h=8..128（当前六档
无差别，说明深窗口无信息）。

附带的诊断信息（与第一轮相同）：AR≈0.988 在**无 tabu 对照**里同样成立，说明「等规模同 Hamming 的走动」
本来就在发生；v10 观察到的「回原解」是**跨周期终态同解**，不是「一步都不动」，缺的是「走动 → +1」的通道
（与 t4 I1/I4 的结论一致）。

**C4 结案（t11 定稿措辞，收窄版）。** 在**同一移动族**且起点**已极大**时，记忆只能重排/收缩候选、**不能创造
+1 移动**；M4（k≤6 精确交换 648 格全 0%）+ 三重不动点（t2 §a）⇒ 该线的**增益上界为 0**。因此 C4/L04
不再花 GPU（不进 1 s 臂），实现与实测保留为可复现的机制证据。
**两点口径必须一起写，否则会被误读**：(1) 这里否掉的是**在既有平台上的增益**，不是「记忆类干预永远无效」——
tabu 当然**能改变搜索路径**（本表 ER=0.0668 就是路径被改的直接证据），它只是无法把路径改到新的 +1 区域；
(2) **AR（0.988）只是机制量**（「轮末解与原解规模相同」的比例），**不是收益量**，不得当成「探索充分」或
「有信号」的证据；收益判据只看 gain 与 `refreshed_rounds`。

---

## 4. 与冻结基线的可比性（门 0）

- **代码可比性**：门 0 (i) 用 `--fixed_steps 808`（≡ `engine.run(inf, ..., max_blocks=ceil(808/10))`）跑 1000 图，与基线同臂、同 K、同 dtype/布局、同 seed、同图序，唯一差异是「用步数而不是墙钟截断」，这正是 §2.7 要求的确定性口径。
- **既有产物零污染**：v11 的逐图 CSV 走 `--v11_csv` 另写，既有 `padded/padded_ref0.csv`、`padded_ref0.json`、summary 的字段与写盘路径未改（`models/eval_v8_padded.py:377` 之后才分支）。
- **时钟口径单独报告**：`--fixed_steps 0` 的 1.0 s 结果标 `steps=`，不混入 A0 对照池（§2.7 (ii)）。
- 图序偏差提醒：`RB_small_author_test` 前 60 图均值明显高于整体（既有 2000 行基线 CSV：first60 = 20.5667 vs 全体 20.0675），所以「跑到 60 图时 20.45」不是漂移信号。

门 0 判据（t11 修订，读数据不读 flag）：

| # | 检查 | 阈值 | 来源 |
| --- | --- | --- | --- |
| 1 | `search_size` 均值 vs A0 参考 **20.0690**（声明配置 fp16+dense） | `\|Δ\| ≤ 0.01`（`CONFIG_SENSITIVITY = 0.007` 只作叙述，**不进容差**） | legacy CSV |
| 2 | 步数 | **恰 810**（硬校验，逐图全命中） | legacy CSV |
| 3 | `dtype` / `dense` / `fp32_csr` | 逐图 **100% 一致**：`fp16` / `1` / `0` | **`--v11_csv` 的行**（不是命令行 flag） |
| 4 | `fixed_steps` | 808（⇒ 810 步） | `--v11_csv` 的行 |
| 5 | `final_size` | **只报告**，注明 ±0.0005 PAVE 预留抖动 | legacy CSV |

门 0 实测（1000 图，GPU1；`outputs/v11/gate0/gates_gate0.json` 是主闸门 (i) 的判决）：

| 口径 | final | search | 步数 | ms/步 | dtype/dense/fp32_csr（**行内**实测） | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| (i) **声明配置** `--mixed_precision fp16 --dense` + 固定 808 步 | 20.0705 | **20.0690** | 810.0 | 0.867 | fp16 / 1 / 0 | **pass（8/8 全绿，exit 0）** |
| (i)′ fp32+dense：**配置偏差**（脚本漏传 `--mixed_precision fp16`，§7.6） | 20.0765 | 20.0760 | 810.0 | 1.22 | fp32 / 1 / 0 | 仅 `dtype_fp16=false` ⇒ 判「配置错」 |
| (ii) 1.0 s 时钟（只报告） | 20.0785 | 20.0775 | 1103.7 | 0.859 | fp16 / 1 / 0 | 不参与判定（`steps_810=False` 属预期） |

A0 判据参考 = **20.0690 @ 810 步**（声明配置 fp16+dense，`REFERENCE_SEARCH`；t16/t13-F1 由 20.0760 更正），容差 `|Δ| ≤ 0.01`、`CONFIG_SENSITIVITY = 0.007`（仅叙述）；
声明配置实测 **20.0690**（Δ = −0.007 ≤ 0.01 ⇒ pass）。`20.0685` 标 `historical_clock_reference_unreproducible`
——但 §7.6 证明**在 fp16+dense 下它与本次 A0 只差 +0.0005（999/1000 图逐位相同）**：不可复现的是**旧 run 的
配置记录**（R5），不是数字本身。对照臂 `outputs/v8_macro/arm_fp16_dense/`：`final = search = 20.0685`、
`pave_gain = 0`、步数 min 725 / mean 808.29 / max 940 / sd 51、ms/步 1.1787。
**逐图配对（声明配置 vs 对照臂）：999/1000 相同，仅 `graph_000466` +0.5（净值 +0.5/1000）**；
973/1000 + 27 行翻牌的那张表属于 fp32 那一跑（§7.5）。

---

## 5. 已知限制

1. **C3 判死不可绕过**：`--loss unroll` 目前只在「单步信用」意义下可训练，梯度对解码大小没有方向信息（§3.4）。这条路径保留为**可复现的负面证据**（`run.json` 里的 `unroll_loss_trace/unroll_size_trace/degraded_reason` 原样保留在 `outputs/v11/train_smoke*/`）。
2. **`--num_tp` 与协议冲突**：冻结训练循环对每张图做 `num_tp` 次更新（默认 100），而 §2.5 写的是「6 epoch、每 epoch 遍历 1000 图」；实测 88 s/图 ⇒ 1000 图 × 1 epoch ≈ 24 h，A1 原协议在 8.2 GPU·h 预算内**跑不完**。若要跑必须显式 `--num_tp 2`（≈3 GPU·h）并当作协议偏离记录。
3. **自适应降级的副作用**：时间门以**第一次 update**的耗时为判据（含 CUDA 预热），因此首次几乎必然 >0.6 s 而降到 128；`measured[0]` 记的是 200 步那次，`steps` 记的是降级后的值，复现时必须看 `requested_steps`。
4. **GPU1 与他人共享**：冒烟期间 GPU1 有约 17.9 GB 他人占用、利用率 99%，因此时钟口径的 ms/步只能同会话比较（§2.7 已按此设计）。
5. **C4 探针的起点是贪心解**（17.1），不是采样器/PAVE 的收敛解（~20.0）；键语义修正后 FIFO 已确实 fire
   （ER=0.0668、命中 4493/63341），起点仍只影响 gain 的绝对值，不影响「记忆不创造 +1 移动」这一结论（§3.6）。
6. **`models/mis_v11_gates.py audit` 模式**已实现但尚未在真实 A1/A1′ 上跑（A1 判死，不再产生 `run.json` 对）。
7. 训练冒烟用的图数很小（2–8 图），只证明「路径不 NaN + 阈值门工作」，不构成质量结论。
8. **配置敏感度 0.007（fp32↔fp16，确定性、可避免、不可平均）**：声明配置（fp16）对三周前基线是
   **999/1000 逐位相同、仅 `graph_000466` +0.5**（§7.6）；那套「973/1000、27 行翻牌、净值 +8/1000=+0.008」
   属于 **fp32 误配跑**（§7.1/§7.5），它度量的是**两条数值路径之间的差**，不是运行间抖动——同一配置重跑
   是 1000/1000 逐位相同（§7.3）。因此配置差异要**靠产物读检判死**（R2），不能靠容差吸收；
   任何 ≤0.007 的均值差都不构成结论，且优先用 `search_size`（`final_size` 另有 ±0.0005 PAVE 抖动，见 #9）。
9. **`final_size` 的 ±0.0005 时钟抖动**：`search_size` 逐位确定，但 PAVE 的 0.2 s 墙钟预留使同一命令
   两次运行的 `final_size` 差 0.0005（A0 vs A0_repeat 实测）。任何 ≤0.001 量级的比较都必须用
   `search_size`，或对齐 `--reserve_frac 0`；v11 的判据（Δ≥+0.020）是它的 40 倍，不受影响。
10. **`--fixed_steps` 与 PAVE 预留的交互**：固定步数口径下 `cutoff=inf`，搜索不再吃 `reserve_frac`，
   PAVE 另拿一份 `perf_counter()+reserve` 的墙钟，因此固定步数臂的 `final_size` 比时钟臂多 0.0005
   （`pave_gain` 0.0005 vs 0）。要让两口径严格可比，应对齐 `--reserve_frac 0`。

---

## 6. 产物清单

```
models/mis_v11_unroll.py            展开器 + 训练钩子 + run.json
models/probe_unroll_grad.py         §2.9 梯度五项自证
models/mis_v11_gates.py             selftest / engine / gate0 / audit
models/mis_v11_tabu.py              C4/L04 动作级 FIFO
models/probe_tabu_walk.py           C4 探针（ER/AR）
train_scripts/v11/{a0_gate0,c3,probe_grad,smoke4}.sh, README.md
outputs/v11/smoke/gates_engine.json          engine 闸门 7/7
outputs/v11/smoke/{timing,fixed}/v11_rows.csv 部署口径 + 固定步数
outputs/v11/probe_unroll_grad.json           20 图探针
outputs/v11/probe_unroll_grad_200.json       200 图探针（判死）
outputs/v11/probe_unroll_grad_200fin.json    最终梯度定义下的 200 图
outputs/v11/train_smoke2/run.json            unroll 训练冒烟（有限 loss + 降级记录）
outputs/v11/tabu_walk_200/{tabu_walk.json,tabu_walk.csv}       C4 探针 键错版（ER≡0，仅作缺陷留档）
outputs/v11/tabu_walk_keyfix/{tabu_walk.json,tabu_walk.csv}   C4 探针 修正版（cross/same × h=0..128，200 图）
outputs/v11/gate0/gates_gate0.json          门 0 (i) 判定 + 逐项 checks
outputs/v11/gate0/a0_steps808_n1000/padded/padded_ref0.csv   1000 图 × 810 步（A0 步数口径原始行）
outputs/v11/gate0/a0_steps808_n1000/v11_rows.csv            同一次 run 的 v11 逐图 CSV
outputs/v11/gate0/a0_clock1s_n1000/                          门 0 (ii) 时钟口径（只报告）
models/summarize_v11.py              配对 + 三 seed pooled 自助统计
outputs/v11/gate0/a0_steps808_n1000/         门 0 (i) 逐图产物
outputs/v11/gate0/a0_clock1s_n1000/          门 0 (ii) 逐图产物 + clock_report
```

## 7. 门 0 实跑结果

> **定稿口径（t16，2026-11-18）**：A0 = **20.0690（search）/ 20.0705（final）**，声明配置
> `--mixed_precision fp16 --dense --fixed_steps 808`（810 步、1000 图、K=200、seed 1017、0.867 ms/步）。
> §7.1–§7.5 记录的是**首跑（fp32 误配：脚本漏传 `--mixed_precision fp16`）**，其中被称作「A0」的
> 20.0760 / 20.0765 **只作 `misconfig_fp32_sensitivity = 0.007` 的证据**，不得再当 A0 使用；
> 声明配置的定稿实测与 999/1000 逐图台账见 §7.6，判据参考见 `models/mis_v11_gates.py`
> （`REFERENCE_SEARCH = 20.0690`、`CONFIG_SENSITIVITY = 0.007`）。

命令：`bash train_scripts/v11/a0_gate0.sh 1`（首跑 12:44:58 起为 fp32 误配；加 `--mixed_precision fp16` 后 21:14 起重跑，见 §7.6），GPU1 单进程。

### 7.1 (i) 步数匹配口径（首跑：fp32 误配，仅作配置敏感度证据）

`--fixed_steps 808` → `max_blocks=ceil(808/10)=81` → **每图 810 步**（`steps` 列全 810，断言成立），1000 图 × repeats 2。

| 指标 | 实测 | 参照 | 差 |
| --- | --- | --- | --- |
| `final_size` 均值 | **20.0765** | 20.0685 | **+0.0080** |
| `search_size` 均值（PAVE 之前） | **20.0760** | 20.0685 | **+0.0075** |
| 平均步数 | 810.0 | 808.29 | +1.71 |
| ms/步 | 1.287 | 1.1787 | +9.2%（机器更慢） |

`gates_gate0.json`：`figures=true / steps=true / dtype_fp16=true / dense=true / not_fp32_csr=true`，
**`solution=false`**（±0.0005 容差下 +0.008 判不过）⇒ 按 §2.7 字面判据，**(i) 不过**。
（这一跑后来被证明是 fp32 误配：`--mixed_precision` 没传，见 §7.6。定稿的声明配置实测是
search **20.0690** / final **20.0705**，八项全绿——本节的 20.0760/20.0765 **只作 `misconfig_fp32_sensitivity`**。）

**这个 +0.008 不是代码漂移**，三条证据：

1. **逐图配对：973/1000 图完全相等**（gap=0.0），18 图 +1（其中 2 图因 `repeats=2` 之一触发 PAVE 呈 +0.5）、9 图 −1，净值 **+8.0 节点 = +0.0080**（逐图台账见 §7.5）。即差异只来自 **27 行图的节点级翻牌（±1 行 25 + 0.5 行 2）**，不是系统性偏移。
2. **步数相同的图 gap 恒为 0.00000**（`810 − steps_base = 0` 的 4 张图），说明同一 (图, seed, 步数)
   下本实现与参照实现**逐位一致**；这与 `gates_engine.json` 的 7 项逐位闸门（eager vs captured、
   unroll vs engine 单步、decode 与 A.3 参照一致）互为印证。
3. 参照臂是**时钟驱动**的：步数随机器负载浮动（min 725 / mean 808.29 / max 940 / sd 51），
   而固定步数臂恒为 810。因此在平台期，27/1000 张图被翻牌：**+8.0 节点 / 1000 = +0.0080**（`final_size`；`search_size` 侧 +0.0075，差额正是那 2 张图的 PAVE 增益）。

**但「步数差」并不能完全解释这 27 行图**，必须如实写明：把每张图按参照步数与 810 对照，
**6 张「我们更好」的图参照跑了 875–910 步（比我们多）**，**6 张「我们更差」的图参照只跑了 755–780 步
（比我们少）** —— 两个方向都违反「archive 对步数单调」，所以「同一 (图, seed, 步数) ⇒ 同一解」
在**跨运行**层面不成立。可能的成因只有两个：
* (a) `outputs/v8_macro/arm_fp16_dense/` 是数周前跑出来的，用的是**当时未提交的工作区**（`git status` 显示
  五个冻结文件至今仍是 `M`，见 §7.4 仓库事实）——「之后零改动、`git diff` 可证」**不成立**；
* (b) fp16 路径下的 ±1 节点级差：**不是随机非确定性**（§7.3 证明今天 1000/1000 逐位相同），而是**数值路径差异
  （解码配置 / fp32↔fp16 等，未单独隔离开）**——§7.4 的三路对照只能证明「引擎层无漂移、dtype 在产物里生效」，
  不能把 ±1 单独归因到解码时刻。

两者都只影响 27/1000 张图的 ±1 节点（净值 +8.0，符号双向：18 上 / 9 下），**不影响 v11 的任何结论**
（v11 的判据是 Δ≥+0.020，即 20 个节点量级）。要钉死是哪一种，做一个 25 min 的实验即可：
**同一条 `--fixed_steps 808` 命令再跑一次**，逐图比较 —— 若两次逐位相同，则今天的引擎是确定性的，
±1 翻牌来自 (a) 旧 run 的驱动/配置；若两次也不同，则 (b) 成立，门 0 的容差必须按「±1 节点翻牌」重写。
结果见 §7.3，驱动层对照见 §7.4，逐图台账见 §7.5。

**建议（需 captain 裁决）**：门 0 (i) 的 ±0.0005 在「参照臂步数随机、固定臂步数恒定」的组态下
不可能满足。两个可选修正：
* **(a) 重设判据（t11 已采纳，见 `models/mis_v11_gates.py`）**：`|Δ search_size| ≤ 0.01`（覆盖 27 行翻牌图的净值上界 + 那 0.007 的 fp32↔fp16 配置敏感度），或
* **(b) 重设基线（推荐）**：把 A0 就是这次 `--fixed_steps 808` 的产物（`outputs/v11/gate0/a0_steps808_n1000/`，
  1000 图 × seed 1017 × 810 步）作为 v11 的 A0 对照——它与未来的 A1/A1′ 用**完全相同的步数与引擎**，
  对照不再受机器速度影响。20.0685 作为「历史时钟口径参照」保留，只做报告。

### 7.2 (ii) 时钟口径（只报告；此臂也是 fp32 误配跑）

同一套 flags 去掉 `--fixed_steps`，1.0 s deadline，1000 图（13:21:19 → 13:55，`outputs/v11/gate0/a0_clock1s_n1000/`）：

| 指标 | (ii) 本机实测 | 参照 `arm_fp16_dense` | 差 |
| --- | --- | --- | --- |
| 解 `final_size` | **20.0725** | 20.0685 | +0.0040 |
| 解 `search_size` | **20.0715** | 20.0685 | +0.0030 |
| 步数 | **736.3** | 808.29 | **−8.9%** |
| ms/步 | **1.285** | 1.1787 | +9.0% |
| load / decode | 6.70 ms / 1.51 ms | — | — |

按 §2.7 的规则：步数差 8.9% **未超过 10% 的「机器速度不同」阈值，但 ms/步 恰好 +9%**，
因此这条口径的解释是「同一台机器、稍慢一点」；该结果**只报告，不进 A0 对照池**。
注意它与 (i) 的对比恰好说明平台有多平：**736 步 20.0715 vs 810 步 20.0760**（差 +0.0045/74 步 ≈ 6e-5/步），
而三周前基线在 808 步上是 20.0685。⇒ 门 0 的 ±0.0005 **低于 fp32↔fp16 的配置敏感度 0.007**，
任何一次跨配置重跑都无法满足；这不是实现问题，是判据标定问题（t11 已把判据改为 `|Δ|≤0.01` +
产物读检，见 §7.6）。

### 7.3 同一命令重跑（确定性判定；两次都是 fp32 误配跑，结论对声明配置同样成立）

`--fixed_steps 808` 同一条命令再跑一次（`outputs/v11/gate0/a0_steps808_repeat/`，14:34 完成）：

| 比较 | 结果 |
| --- | --- |
| run1 (13:21) vs run2 (14:34)，逐图 `search_size` | **1000/1000 完全相同，0 张不同** |
| run1 / run2 / 旧基线 均值 | 20.0760 / 20.0760 / 20.0685 |
| run1 vs 旧基线 | **27 行翻牌**（18 行上升：16 张 +1 + 2 张 +0.5；9 张 −1；净值 +8/1000，见 §7.5） |
| 步数 | 810.0 / 810.0 / 808.29 |

**判定：今天的引擎是逐位确定的**（两次独立进程、相隔 73 min、GPU 上有他人负载，结果完全相同）。
因此 27 行翻牌**不是 fp16 随机噪声**。最初把成因归为「旧基线由更早的引擎构建产出」（依据仅是 mtime：
旧 CSV 写盘 2026-11-16 14:09，而 `models/mis_v8_pad.py` mtime 2026-11-16 20:51，晚 6.7 h），
但 §7.4 的三路对比**否掉了这个归因**：HEAD `6f3b624` 引擎与当前引擎在最小驱动下逐图一致（都 18/27 得旧值），
差异出现在生产二进制（0/27）。因此成因只能记为**驱动/解码配置差异之一（未单独隔离）**：§7.4 证明的是
「引擎层无漂移 + dtype 在产物里生效」，**推不出**「解码时刻就是全部成因」；而旧 run 的命令与工作区
状态从未记录 ⇒ **不可审计**（R6）。净差归属见 §7.4 末段。

**附带量化**：两次运行的 `search_size` 逐位相同，但 `final_size` 差 **−0.0005**（20.0765 vs 20.0760）
——这一档差异全部来自 PAVE 的 0.2 s 墙钟预留（`pave_deadline = perf_counter() + reserve`），
即 **`final_size` 存在 ±0.0005 的时钟抖动，恰好等于门 0 规定的容差**。所以门 0 的 `solution` 项
即使在完全确定性的引擎上也必须用 `search_size`（或对齐 `--reserve_frac 0`）才可能有意义。
`models/summarize_v11.py --a A0 --b A0_repeat` 的输出正是这个：`1 s Δ=−0.0005 CI[−0.0015,+0.0000]`。

**结论（可直接用于裁决）**：
* 门 0 (i) 的 `solution` 项**不可能用旧基线满足**（±0.0005 低于 fp32↔fp16 的配置敏感度 0.007）；
* v11 应采用 **§7.1 建议 (b)**：以本次 `--fixed_steps 808`（810 步）1000 图结果作为 **A0 步数口径基线**
  （`outputs/v11/gate0/a0_steps808_n1000/`，已由 run2 复现验证），20.0685 作为
  `historical_clock_reference_unreproducible`（历史时钟口径参照）只做报告；
* 这条基线同时具备**完全确定性**（run1 = run2），比旧基线更适合作为 A1/A1′ 的对照。

### 7.4 驱动层对照：只回答「引擎有没有漂移」，不定因（HEAD 构建实测 + 仓库事实）

**请求**：captain m01062 要求用 HEAD `6f3b624` 的 `mis_v8_pad.py`/`models.py` 跑同一批图，以区分
「当前文件语义漂移」与「旧 CSV 是第三个中间态」。

**方法**：服务器是纯同步目录、**不是 git 仓库**（`git worktree add` → `fatal: 不是 git 仓库`），故用文件级覆盖：
本地 `git show 6f3b624:models/{mis_v8_pad,models}.py` → `tmp/head_build/models_head/`（279 / 124 行），
上传后以 `PYTHONPATH=tmp/head_build/models_head:models` 覆盖导入（`inspect.signature` 确认 HEAD 版无 `dense`、
无 `sync_each_block` 形参）。驱动 `tmp/head_flips.py` 复用生产解码路径（`policy_parser` / `load_policy` /
`dataset_at(official=False)` → `decode_top` → `decode_gradient_flip` → `np.packbits`，`max_blocks=81`），
在 27 行翻牌图 + 8 张对照上重算。

**仓库事实（R6，必须与结论一起读）**：

| 事实 | 证据 |
| --- | --- |
| 冻结协议里的 `models/{main,models,solvers,mis_v8_pad,eval_v8_padded}.py` **五个都是未提交的工作区改动** | `git status --short` → `M models/{...}.py`；HEAD = `18cb409`（2026-11-15 16:46:52），最后一次触及 `mis_v8_pad.py` 的提交是 `6f3b624`（2026-11-15 15:56:18） |
| 因此「git diff 可证零改动」**不成立** | 本次 additive 改动与旧 run 用的未提交改动**在同一份工作区里不可区分** |
| `models/data/`（RB 图集）**不入库** | `.gitignore` 排除；只在服务器 / 本地各有副本 |
| **没有 checkpoint 快照** | `*.pt/*.pth/*.ckpt` 被 `.gitignore` 排除，全仓 `find` 命中 0；冻结骨干与 PAVE 权重只在服务器 `outputs/v2/none_seed0/checkpoints/`（`best.pt` mtime 11-08 11:05） |

⇒ 「旧基线构建」这个说法本身不可证：旧 run 用的是**未记录的未提交工作区**。只能记为「**旧 run 配置不可考**（R5）」
与「fp32↔fp16 配置敏感度 0.007」两件事，不能记为「某个 commit 的构建差异」。

**实测**：

| 驱动 / 引擎 | 27 行翻牌里得「旧值」 | 说明 |
| --- | --- | --- |
| HEAD 引擎（`6f3b624`）+ 最小驱动 | 18/27 | 其余得新值；8 张对照全同（如 `graph_000000 = 22`） |
| 当前引擎 + 最小驱动 | 18/27 | 与 HEAD 逐图一致 ⇒ **引擎构建层没有语义漂移** |
| 当前引擎 + 生产 eval 二进制（`--fixed_steps 808`，`--v11_csv`） | 0/27 | `graph_000063 = 22`、`64 = 23`、`77 = 21`，与 §7.1 的 gate0 CSV 逐位一致 |

**这组实验能推出什么 / 推不出什么**：HEAD 引擎与当前引擎在最小驱动下逐图一致（都 18/27 得旧值），差异只出现在
生产二进制（0/27）⇒ **能推出**「引擎构建层无语义漂移」与「dtype 确实在产物里生效」（R2 的读产物检查）。
**推不出**「解码时刻是全部成因」：最小驱动与生产二进制在解码发生点上确实不同（一次喂完 81 块再解码 vs `run()`
在末块之后沿 `pending` 路径解码），但这条差异与 fp32↔fp16 的数值路径差**叠在一起、本轮没有单独隔离**
（隔离需固定 dtype、只改解码时机各跑一次 1000 图，未做）。所以本节只作驱动层对照，成因记为
「配置 / 驱动差异，未单独隔离」。

**净差归属（27 行翻牌）**：净差 **+8.0/1000 = 20.0765 − 20.0685** 由 **fp32 误配**解释——声明配置（fp16）与旧
20.0685 是 **999/1000 逐图相同**（唯一差异 `graph_000466` 19.5→20.0，由步数 775→810 解释；mean_search +0.0005、
mean_final +0.0020）；fp32 误配跑与旧的差是 **973 相同 / 18 行上升（16 张 +1 + 2 张 +0.5）/ 9 张 −1 = +8.0 节点
= +0.0080**。即「±1 行数 25、+0.5 行数 2、合计 27 行」，与 §7.5 台账逐行可对账。

**术语（t16 定稿，全文统一）**：只用 **「配置敏感度 0.007（fp32↔fp16，确定性、可避免、不可平均）」**；
「噪声底」「离散翻转分辨率 0.008」两个说法**作废**——它们把一次确定性的配置差异讲成了随机抖动，机制说错了。
理由：同一配置两次运行 **1000/1000 逐位相同**（§7.3），所以不存在「运行间分辨率」；而 fp32 与 fp16 之间的
0.007（§7.6 的 973/1000、27 行翻牌那套）是**换了一条数值路径**，换回去就消失，因此它可避免、
也不能靠平均或容差消掉 ⇒ 由 R2 的产物读检直接判「配置错」。
对外口径：1.0 s 档 **20.0690** 对 RLSA 的 1 s 插值 ≈20.084 **落后 0.015（≈2×配置敏感度）**；
本轮目标 +0.025 → **20.094** 才算明确领先。

### 7.5 fp32 误配 vs 声明配置的翻牌台账（27 行 = ±1 行 25 + 0.5 行 2；m01352 要求完全对账）

> 本节的 27 行属于 **fp32 误配跑**（`--mixed_precision` 漏传）。声明配置（fp16）对同一旧基线的台账是
> **999/1000 逐位相同、仅 `graph_000466` 19.5→20.0**（§7.6）；本节保留在文档里，正是为了给人看
> 「换一条数值路径能把均值整体平移 +0.008」这件事有多大。

来源：服务器 `tmp/ledger.py` 逐图 join 两份 CSV ——
旧 `outputs/v8_macro/arm_fp16_dense/padded/padded_ref0.csv` vs 新 `outputs/v11/gate0/a0_steps808_n1000/padded/padded_ref0.csv`。
数值是**两次 repeat 的均值**，故出现 .5。

**解码发生点**：两臂都是 `--decode_every 1000`、每臂块数 ≤ 94 ⇒ 全程只解码一次、发生在末块之后 ⇒
**decode step = 该臂总步数**（新臂恒 810，旧臂见表）。
**PAVE 预留**：两臂都为每张图分配 `reserve`（0.05 s 墙钟）的 PAVE 窗口；新臂因固定 810 步 ≈1.04 s 超出
1.0 s 名义预算，27 行的 `deadline_overrun` 全为 1，但 PAVE 仍拿到自己的 `perf_counter()+reserve` 窗口。
「PAVE 是否改了值」一列指该图最终尺寸是否因 PAVE 而变。

| graph_id | 旧 size | 新 size | Δ | decode step（新） | 旧步数 | PAVE 改了值？ |
| --- | --- | --- | --- | --- | --- | --- |
| graph_000064 | 22.0 | 23.0 | +1 | 810 | 910 | 否（仅分配预留） |
| graph_000077 | 20.0 | 21.0 | +1 | 810 | 775 | 否 |
| graph_000175 | 21.0 | 22.0 | +1 | 810 | 780 | 否 |
| graph_000243 | 19.0 | 20.0 | +1 | 810 | 875 | 否 |
| graph_000269 | 18.0 | 19.0 | +1 | 810 | 795 | 否 |
| graph_000339 | 22.0 | 23.0 | +1 | 810 | 765 | 否 |
| graph_000424 | 18.0 | 18.5 | +0.5 | 810 | 875 | **是**（2 次 repeat 中 1 次 +1） |
| graph_000466 | 19.5 | 20.0 | +0.5 | 810 | 775 | **是**（同上） |
| graph_000541 | 22.0 | 23.0 | +1 | 810 | 760 | 否 |
| graph_000548 | 21.0 | 22.0 | +1 | 810 | 760 | 否 |
| graph_000562 | 19.0 | 20.0 | +1 | 810 | 760 | 否 |
| graph_000611 | 22.0 | 23.0 | +1 | 810 | 760 | 否 |
| graph_000666 | 21.0 | 22.0 | +1 | 810 | 765 | 否 |
| graph_000764 | 19.0 | 20.0 | +1 | 810 | 885 | 否 |
| graph_000857 | 21.0 | 22.0 | +1 | 810 | 885 | 否 |
| graph_000924 | 21.0 | 22.0 | +1 | 810 | 905 | 否 |
| graph_000927 | 21.0 | 22.0 | +1 | 810 | 885 | 否 |
| graph_000940 | 19.0 | 20.0 | +1 | 810 | 760 | 否 |
| graph_000063 | 23.0 | 22.0 | −1 | 810 | 895 | 否 |
| graph_000189 | 20.0 | 19.0 | −1 | 810 | 765 | 否 |
| graph_000367 | 23.0 | 22.0 | −1 | 810 | 770 | 否 |
| graph_000385 | 23.0 | 22.0 | −1 | 810 | 770 | 否 |
| graph_000429 | 23.0 | 22.0 | −1 | 810 | 775 | 否 |
| graph_000447 | 20.0 | 19.0 | −1 | 810 | 780 | 否 |
| graph_000589 | 22.0 | 21.0 | −1 | 810 | 880 | 否 |
| graph_000625 | 23.0 | 22.0 | −1 | 810 | 755 | 否 |
| graph_000858 | 21.0 | 20.0 | −1 | 810 | 905 | 否 |

**对账（0.008 的均值跳变被这张表完全吃掉）**：

```
identical 973 × 0                =   0.0 nodes
plus  18: 16 × (+1) + 2 × (+0.5) = +17.0 nodes
minus  9: 9 × (−1)               =  −9.0 nodes
------------------------------------------------
net                                 +8.0 nodes / 1000 graphs = +0.0080
实测 Δmean(final_size) = 20.0765 − 20.0685 = +0.0080  ✓ 逐位吻合
```

`search_size` 侧同量：`20.0760 − 20.0685 = +0.0075`。（本节原写「差额 +0.0005 正是那 2 张图的 PAVE 增益」，
已被 §7.5.1 的重建推翻：重建跑里 424/466 的 `pave_gain` 全为 0，`search_size` 与 `final_size` 两侧计数相同。）

**读法（三条，别再改口径）**：
1. 27 行 = `plus 18 行（16 张 +1 + 2 张 +0.5）+ minus 9 行`，**双向**都有 ⇒ 不是系统性变好或变差，
   是边缘节点在数值路径边界上的翻转（具体是哪条路径，未单独隔离）；
2. 翻牌与旧臂步数**没有单调关系**（旧步数最大的 910/905/905/895 四张里既有 +1 也有 −1）⇒ 促因不是步数预算，
   而是驱动/解码配置差异之一（未单独隔离，§7.4）以及 PAVE 的墙钟时序；
3. 因此门 0 / 所有配对比较一律用 `search_size`，`final_size` 只报告且必须标注 ±0.0005 的 PAVE 抖动。

#### 7.5.1 重建 fp32 artifact 的逐图对账（t20）：26/27，`graph_000424` 未复现

- **artifact**：服务器 `outputs/v11/gate0/a0_misconfig_fp32_repro/`（`padded/padded_ref0.csv` 278 928 B、
  `v11_rows.csv`、`padded/padded_ref0.json`：`graphs=1000, repeats=2, chunk=10, capture_s=0.9525,
  load_s=0.0082, wall_time_s=2172.66`），2026-11-18 00:28 落盘；配置与首跑同
  （`--mixed_precision no --dense --fixed_steps 808`，不传 `--split`），GPU1 单进程，驱动 `tmp/t20_fp32_repro.sh`。
- **均值**：`mean_final 20.0685 → 20.0760`、`mean_search 20.0685 → 20.0760`（首跑 fp32 口径为 20.0760/20.0765）、
  `steps 808.29 → 810.00` ⇒ 判据「20.0760 ± 0.01」**恰好命中**。
- **逐图计数**：`identical 974 / +1.0 16 / +0.5 1 / −1.0 9`，净 **+7.5 nodes = +0.0075**；
  `search_size` 与 `final_size` 两侧计数完全相同（都是 974/16/1/9）⇒ 这批翻牌不经过 PAVE。
- **与本节 27 行表逐行核：26/27 一致**；唯一不符 `graph_000424`（本表 18.0→18.5，重建 18.0→18.0）。
- **机制更正**：`pave_gain` 在 424/466 四行全为 0，重建跑内也没有同图 repeats 不一致。
  `graph_000466` 的 +0.5 **已定因**：旧基线两次 repeat 步数不同（`20 @790 步` 与 `19 @760 步`，均值 19.5），
  重建跑两次都是 `[20,20]` ⇒ 差 0.5 来自旧时钟臂的步数差跨过尺寸边界，与本表 497 行写的
  「PAVE 改了值？= **是**」无关（PAVE 未改值）。
  `graph_000424` 的 +0.5 **无法定因**：该差值的来源 CSV（`outputs/v11/gate0/a0_steps808_n1000/`）已被两次 fp16
  重跑覆盖，只存于本表；重建跑里 424 两次 repeat 都是 18.0（旧臂 18 @870 / 18 @880）⇒ 差 0。
  故 496/497 行「PAVE 改了值？= 是」这一解释整体**不成立**。
- **结论（待队长裁定，未改判据与 A0）**：均值口径在容差内，但「973 相同 / 18 上升（16 张 +1 + 2 张 +0.5）/
  9 下降 / 净 +8.0」这组逐图数字**不能逐位重建**（重建为 974 / 16 张 +1 + 1 张 +0.5 / 9 / 净 +7.5，
  差额恰是 `graph_000424` 的 0.5 节点 = 0.0005 均值）。故 27 行台账只作「量级与结构」证据；
  `final_size` 的 ±0.5 行不作逐位复现判据（与本节读法 3「final 只报告」一致）。

### 7.6 配置缺陷的发现与修正：R2 的「读产物」检查第一次就命中了真问题

**事实链**：

1. 第一次全量门 0（17:56–18:31）用的旧 `a0_gate0.sh` 的 `COMMON` 里**只有 `--dense`、没有 `--mixed_precision fp16`**
   ⇒ 实际跑的是 **fp32 + dense**。日志看起来完全正常（1.22 ms/步、810 步、search 20.0760），
   任何「拿 flag 跟自己的期望值比」的闸门都会判 pass。
2. t11 的 R2 把 dtype / dense 改成从 **`--v11_csv` 的每一行**读：`dtype=fp32` ⇒ `dtype_fp16=false`，
   `gate0` **exit 3**，主闸门判「配置错、不许开 A1」。
3. 加 `--mixed_precision fp16` 重跑（21:14–22:17）⇒ **8/8 全绿、exit 0**。

**数值后果（27 行翻牌的净差归属：fp32 误配）**：

| 运行 | 配置 | search | final | 步数 | ms/步 | 与对照臂逐图比较 |
| --- | --- | --- | --- | --- | --- | --- |
| 第一次（17:56–18:31） | fp32 + dense | 20.0760 | 20.0765 | 810.0 | 1.22 | 973/1000 相同，27 行翻牌，净值 +8/1000 |
| 第二次（21:14–22:17） | **fp16 + dense（声明配置）** | **20.0690** | **20.0705** | 810.0 | 0.867 | **999/1000 相同，仅 `graph_000466` 19.5→20.0** |
| 对照臂（三周前） | fp16 + dense | 20.0685 | 20.0685 | 808.29（时钟） | 1.1787 | — |

⇒ 声明配置把旧基线**复现到 +0.0005（search）/ +0.0020（final）**；先前的 27 行翻牌、+0.008 全部来自
**fp32 与 fp16 的配置差异（配置敏感度 0.007）**，而不是「旧 run 配置不可考」本身，也不是运行间随机性
（同一配置两次运行 1000/1000 逐位相同，§7.3）。

**三件必须一起写清的事**：

1. **两类差异要分开**：同一配置的**运行间**抖动 = 0（1000/1000 逐位相同，§7.3）；**fp32 与 fp16 之间**的差 =
   **配置敏感度 0.007**（确定性、可避免）。论文里只写后者，并且写「按产物判配置」而不是「靠容差吸收」。
2. **20.0685 不是「不可复现的数字」**：在声明配置下它可复现（+0.0005）；不可复现/不可审计的是**旧 run 的
   实验记录**（命令、工作区状态、dtype 都没留）——这正是 R5 的问题。
3. **判据要按声明配置定**：A0 参考 = **20.0690**（t16 定稿，`REFERENCE_SEARCH`）±0.01。20.0760（fp32）
   **不再是参考**——参考偏 +0.007 叠 ±0.01 给出 [20.0660, 20.0860]，而正确带是 [20.0590, 20.0790]，
   会让 20.080–20.086 的真实退化假通过（t13-F1）。配置差异改由 R2 的产物读检判死，不再靠容差吸收。

**Runbook 教训**：门 0 的配置检查一律从**产物**读（已实现）；`--mixed_precision` 这类「默认值就是错误值」的
参数，是「日志看着正常、数值悄悄换了路径」的典型来源。

---

## 8. t11 交付摘要（判据修复 round）

| 项 | 状态 | 证据 |
| --- | --- | --- |
| 门 0 重定基（R7 → t16 定稿） | ✅ | `REFERENCE_SEARCH=20.0690`（声明配置 fp16+dense）、`\|Δ\|≤0.01`、`CONFIG_SENSITIVITY=0.007`（仅叙述）、`HISTORICAL_CLOCK_REFERENCE=20.0685` + `historical_clock_reference_unreproducible` |
| dtype/dense 实检（R2） | ✅ | `eval_v8_padded.py:377-400` 增 `dtype/dense/fp32_csr`；`mis_v11_gates.py::read_artifact` 逐行读；负例自检 `shift_fails/config_fails` 均 true |
| signal-only 降级（R1） | ✅ | `summarize_v11.py` verdict 优先级 `unstable > fail > signal-only > pass`，S1 只认 `pass` |
| selftest 第 10 项 | ✅ | `frozen_equivalence`（`platform_search_tabu(fifo=None)` ≡ 冻结 `platform_search`，2 seed × 3 图） |
| 门 0 跑一次 | ✅ | `bash train_scripts/v11/a0_gate0.sh 1` → 8/8 绿、exit 0；`outputs/v11/gate0/gates_gate0.json` |
| C4 结案 | ✅ | §3.6 末段（收窄措辞 + AR 仅机制量） |
| 文档口径 | ✅ | `README.md` KPI 块；本文件 §0/§4/§5/§7.4/§7.5/§7.6 |

---

## 9. t16 交付摘要（A0 定稿 20.0690 + 判据层/术语同步）

t13 复审给的 F1–F6 一次做完，判据常量、脚本、README、本文件、`train_scripts/v11/README.md` 同时改齐。

| 项 | 改哪里 | 改成什么 |
| --- | --- | --- |
| **F1** 判据层 | `models/mis_v11_gates.py:75` | `REFERENCE_SEARCH = 20.0690`（声明配置 fp16+dense、810 步） |
| | `models/mis_v11_gates.py:77` | 删 `NOISE_FLOOR`，新增 `CONFIG_SENSITIVITY = 0.007`（fp32↔fp16，仅叙述、**不进容差**） |
| | `models/mis_v11_gates.py:499-503` | `reference_kind='A0_declared_config_fp16_dense_810_steps'`、`config_sensitivity` + `config_sensitivity_note='fp32<->fp16, deterministic, not in the tolerance'` |
| | `models/mis_v11_gates.py:26-28` | 模块 docstring 同步（20.0690；配置敏感度只作背景） |
| **F2** 数字 | 本文件 `:14, 247, 250, 315, 335, 455, 543, 557` 与 `train_scripts/v11/a0_gate0.sh:6` | 当 A0 用的 20.0760 → **20.0690**；20.0760/20.0765 只保留在 fp32 误配语境（§7.1/§7.2/§7.3/§7.5/§7.6） |
| **F3** 「逐字节」（旧措辞，已删） | 本文件 `:11, :59`、`train_scripts/v11/README.md:50` | 改「既有路径未改动，未提交状态见 §7.4」/「行为等价」 |
| **F4** 术语 | `README.md:36` 段 + 本文件十余处 | 统一「配置敏感度 0.007（fp32↔fp16，确定性、可避免、不可平均）」；「噪声底」「离散翻转分辨率 0.008」全文只在本节与 §7.4 保留为**作废声明** |
| **F5** README KPI | `README.md:28-38` | KaMIS 改 **20.1020 @ 3.046 s**（并注 `--time_limit=1` 只 99/1000 图返回、那 99 图 19.8384、墙钟地板 1.87 s）；RLSA 定稿句「落后 0.015（≈2×配置敏感度），目标 +0.025 → 20.094 才算明确领先」 |
| **F6** summarize 可跑 | `models/summarize_v11.py:29-55, :93-104` | `paired_bootstrap` 改成逐 op 拷贝（同 dtype/同 generator/同 resample 矩阵），`--check_reference` 与参考实现逐位比对（服务器 `identical=true`，5 个用例）；本机不再需要 `tqdm`/`torch_scatter` |

### 9.1 顺带修掉的一个实质缺陷（R1 的数据路径）

`models/summarize_v11.py:65` 原读 `row.get("seed", 0)`，而 `--v11_csv` 的列名是 **`search_seed`** ⇒ 所有行都落到
`seed=0`，三 seed 稳健性条款**永远不会触发**：一个「3 seed 里 1 个为正」的结果会被报成 `1/1 pass` 并计入 S1。
改为 `row.get("seed") or row.get("search_seed")`，并用合成表验了四档判读（3 seed × 30 图，budget 1.0）：

| 构造（Δ per seed） | 旧代码 | 新代码 |
| --- | --- | --- |
| 1017:+0.05 1019:+0.05 1021:+0.05 | pass（1/1） | **pass（3/3）** |
| 1017:+0.10 1019:0 1021:0 | **pass（1/1）** | **signal-only（1/3）**，不进 S1 |
| 1017:+0.10 1019:−0.05 1021:+0.10 | pass | **unstable**（任一 seed < −0.010） |
| 全 0 | fail | fail |

### 9.2 顺带修掉的一个工程问题（会被推到服务器）

本地写文件时按 Windows 换行落盘，`.gitattributes` 要求 `*.sh/*.py/*.md text eol=lf`：11 个文件（含
`train_scripts/v11/a0_gate0.sh`）变成 CRLF，`bash -n` 直接报 `syntax error near unexpected token '||'`
（`\` 后跟 CR 把续行截断）。已全部归一为 LF，服务器 `file` 确认「UTF-8 text executable 无 CRLF」，
`bash -n` 通过。

### 9.3 改完后的复跑（A0 判据换成 20.0690 之后）

`bash train_scripts/v11/a0_gate0.sh 1`（22:38:42 起，GPU1）：step 0 的 `selftest` 10/10 与
`gate0 --negative-selftest` 三项全 true；**(i) 23:03:49 结束、exit 0、8/8 全绿**：

| 字段 | 值 |
| --- | --- |
| `mean_search_size` / `delta_search` | **20.069** / **0.0**（`reference=20.069`，`tolerance=0.01`） |
| `reference_kind` | `A0_declared_config_fp16_dense_810_steps` |
| `config_sensitivity` | 0.007（`config_sensitivity_note='fp32<->fp16, deterministic, not in the tolerance'`） |
| `mean_final_size` | 20.0705（`final_size_note` 写明只报告、含 PAVE ±0.0005） |
| `mean_steps` / `mean_ms_per_step` | 810.0 / 0.877（上次 0.867，同配置不同负载；固定步数口径下不影响可比性） |
| `historical_clock_reference` / `delta_vs_historical` | 20.0685 / +0.0005（`historical_clock_reference_unreproducible`） |
| `artifact` | `dtype=fp16 dense=1 fp32_csr=0 fixed_steps=808 steps=810`，2000 行 100 % 一致 |

⇒ 判据换成 20.0690 后**既没有变松也没有变严**：clean 产物仍是 `delta=0.0` 全绿，负例（+0.05 位移、fp32 配置）
仍分别被 `search` 与三项产物检查判死，`selftest` 仍 10/10。

**(ii) 时钟臂（只报告，23:37:09 结束）**：`mean_search_size=20.077`（Δ +0.008，`reference=20.069`）、
`mean_final_size=20.0785`、`mean_steps=1079.8`、`mean_ms_per_step=0.878`；`steps_810=false` /
`artifact_consistent=false` 属预期，`passed=false` 不参与判定。JSON 里已无 `noise_floor` 键
（`config_sensitivity=0.007`、`reference_kind='A0_declared_config_fp16_dense_810_steps'`），
脚本末行打印 `=== 门 0 判定：主闸门 (i) exit=0 ===`。
