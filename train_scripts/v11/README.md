# v11–v13 启动器

## 四档规模的 200 ms 主表

RB-small 列沿用已有同图同种子结果。其余三列使用作者原图，分别传入 `RB_large`、`ER_small`、`ER_large`。每次对完整测试集运行种子 1017、1019、1021，同一张卡串行运行不同求解器。下面以 ER-[700–800] 为例，命令从项目根目录执行。

```bash
split=ER_small
out=outputs/v11/scale_deadline_20261120
python tools/prepare_author_scale.py "$split"
bash train_scripts/v11/paper_scale_fused.sh "$split" "$out/$split/fused" 1
for arm in author_rlnn rlsa fld_eg diffuco ltft difusco dgl_treesearch intel ppo_lwd isco_full_window isco author_rlnn_matched; do
  CUDA_VISIBLE_DEVICES=0 bash train_scripts/v11/paper_scale_references.sh "$split" "$arm" "$out"
done
for arm in online_mis redumis; do
  bash train_scripts/v11/scale_kamis.sh "$split" "$arm" "$out/$split/$arm"
done
```

融合与各学习基线在新增三列沿用小 RB 的推理设置和权重。`author_rlnn_matched` 另用作者为大 RB 或 ER 发布的权重。KaMIS 两种可执行程序及外部学习方法的环境见下文。汇总入口会检查各档图编号与种子是否齐全，再从逐图结果重建均值和表格。 [数据集](../../docs/数据集.md) · [汇总脚本](../../tools/summarize_scale_deadline.py)

## 外部基线 200 ms 对照

`models/eval_reference_deadline.py` 按 `idx_0` 至 `idx_999` 读取作者图，对每张图依次运行种子 1017、1019、1021。iSCO 使用 DISCS 的 MIS 模型和自适应 Path Auxiliary Sampler，保留截止前直接采到的合法解。RLSA 使用作者 RB 设置的 200 粒子和 300 步降温日程，在中段和末次调用 A.3 解码。KaMIS 使用单线程 ReduMIS，给内部搜索 180 ms，外层 200 ms 包含启动、读图和收集解。图缓存读取和共享输入转换在逐图计时外。

```bash
out="outputs/v11/reference_deadline_$(date +%Y%m%d_%H%M%S)"
discs="tmp/discs_for_deadline"
mkdir -p "$discs" "$out"
cp -a reference_methods/iSCO/source/discs "$discs/"
python train_scripts/v11/prepare_discs.py "$discs"
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/$discs"
"$HOME/envs/rlnn_mamba/bin/python" -u models/eval_reference_deadline.py isco --output "$out/isco_full_window" 2>&1 | tee "$out/isco_full_window.log"
"$HOME/envs/rlnn/bin/python" -u models/eval_reference_deadline.py rlsa --output "$out/rlsa" 2>&1 | tee "$out/rlsa.log"
"$HOME/envs/rlnn/bin/python" -u models/eval_reference_deadline.py kamis --output "$out/kamis" 2>&1 | tee "$out/kamis.log"
cmake --build tmp/kamis_build --target online_mis --parallel 4
"$HOME/envs/rlnn/bin/python" -u models/eval_reference_deadline.py kamis --kamis tmp/kamis_build/online_mis \
  --output "$out/kamis_online" 2>&1 | tee "$out/kamis_online.log"
"$HOME/envs/rlnn_mamba/bin/python" -u models/eval_reference_deadline.py isco --isco-max-steps 300 --output "$out/isco" 2>&1 | tee "$out/isco.log"
python tools/map_reference_solution_nodes.py models/data/MIS_tensors/RB_small_author_test/tensor_manifest.json "$out/graph_nodes.jsonl"
python tools/summarize_reference_deadline.py "$out" --fused outputs/v11/fused_pave_02s_deadline --output "$out/summary.json"
```

`solutions.jsonl` 保存严格改善的按时合法解，节点用零起点图缓存索引。`graph_nodes.jsonl` 给出每张图的索引到作者节点 ID 的映射。`per_graph.csv` 记录首次及最终计分解可用时刻、总墙钟和截止覆盖。

作者 RLNN 与 FLD-EG 复跑沿用这个入口。作者 RLNN 使用发布的 RB-small ERDOES 权重、FP16、20 粒子、100 步上限及 A.3 解码。FLD-EG 用 PyTorch 重建论文算法 1 和表 3 的 RB-small 设置，使用 FP32 和逐链贪心解码。两组均在中段与末次解码，按实际可用时刻计分。

```bash
out="outputs/v11/reference_deadline_followup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out"
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1
for method in author_rlnn fld_eg; do
  "$HOME/envs/rlnn/bin/python" -u models/eval_reference_deadline.py "$method" \
    --output "$out/$method" 2>&1 | tee "$out/$method.log"
done
python tools/summarize_reference_deadline.py outputs/v11/reference_deadline_20261119 \
  --follow-up "$out" --fused outputs/v11/fused_pave_02s_deadline --output "$out/summary.json"
```

### DiffUCO 与 LTFT

DiffUCO 使用官方发布的 SDDS 对照权重 `m3h9mz5g`，每次从 8 条轨迹中经官方 MIS 条件期望解码选解。限时封装先运行 1 倍步数，剩余时间允许时再运行 3 倍步数，两次解码都计入墙钟。LTFT 用官方 Forward-Looking GFlowNet 训练代码和 RB 小图配置训练 20 轮，再对每张图采样 20 条轨迹。200 ms 内已经同步到主机的最大合法集合参与计分。 [DiffUCO 权重](https://github.com/ml-jku/DIffUCO/blob/main/README.markdown#model-weights) · [LTFT 官方实现](https://github.com/zdhNarsil/GFlowNet-CombOpt)

在实验服务器上使用 JAX 0.6.2、Flax 0.10.4、Jraph 0.0.6.dev0 和 GPU 版 DGL 2.4.0。`diffuco` 和 `ltft` 路径指向两个上游工作副本，`weights` 指向下载的 DiffUCO 权重。CUDA 设备 0 用于 LTFT 训练，设备 1 用于两个方法的计时。训练和验证图由 LTFT 官方生成器使用种子 0 和 10000 分别生成。

```bash
out="outputs/v11/learned_deadline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out"
python train_scripts/v11/prepare_diffuco.py "$diffuco"
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 \
  PYTHONPATH="$diffuco" python -u models/eval_reference_deadline.py diffuco \
  --checkpoint "$weights" --output "$out/diffuco" 2>&1 | tee "$out/diffuco.log"
(
  cd "$ltft/data"
  python rbgraph_generator.py --num_graph 4000 --seed 0 --save_dir rb200-300/train
  python rbgraph_generator.py --num_graph 500 --seed 10000 --save_dir rb200-300/test
)
DGLBACKEND=pytorch CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -u train_scripts/v11/train_ltft.py "$ltft" --output "$out/ltft_training" \
  2>&1 | tee "$out/ltft_train.log"
DGLBACKEND=pytorch CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTHONPATH="$ltft/gflownet" python -u models/eval_reference_deadline.py ltft \
  --checkpoint "$out/ltft_training/alg.pt" --ltft-config "$out/ltft_training/train_config.yaml" \
  --output "$out/ltft" 2>&1 | tee "$out/ltft.log"
python tools/summarize_reference_deadline.py outputs/v11/reference_deadline_20261119 \
  --follow-up outputs/v11/reference_deadline_20261119_followup --learned "$out" \
  --fused outputs/v11/fused_pave_02s_deadline --output "$out/summary.json"
```

### DIFUSCO 与 DGL-TreeSearch

DIFUSCO 采用官方 SATLIB MIS 分类扩散权重和贪心解码，在 RB 小图上做迁移评测。推理步数仅由最大边数图的预热耗时决定。DGL-TreeSearch 用官方 GCN、Hindsight Loss 和单工作进程队列。用 KaMIS OnlineMIS 给独立生成的 4000 张训练图标注，每图内部时间上限 250 ms。DGL 上游参数设为 `epochs=20`，运行第 0 至 20 轮，启用自环。训练中以等价的稠密归一化邻接乘法计算官方图卷积，评测使用官方 DGL 图卷积。推理分支数按预热耗时选定，每个概率图处理结束时记录已得到的完整或部分合法集。 [DIFUSCO 官方发布](https://github.com/Edward-Sun/DIFUSCO#pretrained-checkpoints) · [DGL-TreeSearch 官方实现](https://github.com/MaxiBoether/mis-benchmark-framework#dgl-treesearch)

`difusco` 和 `dgl_search` 分别指向 DIFUSCO 仓库工作副本和 DGL 仓库工作副本的 `solvers/dgl_treesearch` 目录。`rb_train` 指向用上一节的官方 RB 生成器产生的 4000 张训练图。两组均用设备 1 评测。

```bash
out="outputs/v11/supervised_deadline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python train_scripts/v11/prepare_supervised_baseline.py difusco "$difusco/difusco"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$difusco" python -u models/eval_reference_deadline.py difusco \
  --checkpoint "$satlib_weights" --output "$out/difusco" 2>&1 | tee "$out/difusco.log"
python train_scripts/v11/label_dgl_rb.py "$rb_train" tmp/kamis_build/online_mis "$out/dgl_labels/train"
python train_scripts/v11/build_dgl_rb.py "$out/dgl_labels/train" "$out/dgl_labels/graphs_unweighted.dgl"
DGLBACKEND=pytorch CUDA_VISIBLE_DEVICES=0 python -u train_scripts/v11/train_dgl_treesearch.py \
  "$dgl_search" "$out/dgl_labels" "$out/dgl_training" --dense-kernels 2>&1 | tee "$out/dgl_train.log"
checkpoint=$(find "$out/dgl_training" -name '*_final_model32.torch')
DGLBACKEND=pytorch CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$(dirname "$dgl_search")" python -u models/eval_reference_deadline.py dgl_treesearch \
  --checkpoint "$checkpoint" --output "$out/dgl_treesearch" 2>&1 | tee "$out/dgl.log"
python tools/summarize_reference_deadline.py outputs/v11/reference_deadline_20261119 \
  --follow-up outputs/v11/reference_deadline_20261119_followup --learned outputs/v11/learned_deadline_20261119 \
  --supervised "$out" --fused outputs/v11/fused_pave_02s_deadline --output "$out/summary.json"
```

### INTEL 与 PPO-LwD

INTEL 用作者发布的 20 层 GCN 权重，在 RTX 5090 上以 FP32 计算 32 张概率图。导出脚本同时保存原 TensorFlow 计算图对给定输入的预测，GPU 入口据此核对输出。逐图计时包括子图归一化、推理、随机前缀树搜索及节点集交付。使用单工作进程，关闭上游可选的图规约与局部搜索。PPO-LwD 用作者的策略、MIS 环境和 PPO 更新在独立 RB 图上重训。训练集 4,000 图，训练两轮，每 25 次更新在固定的 64 张独立验证图上用 10 条轨迹选权。评测按官方配置并行采样 10 条轨迹，每条最多 32 步，两组均保留截止前真实交付的部分或完整合法集。

```bash
out="outputs/v11/intel_ppo_deadline_$(date +%Y%m%d_%H%M%S)"
work="$PWD/tmp/intel_ppo"
mkdir -p "$work" "$out"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$HOME/envs/nexco/bin/python" train_scripts/v11/export_intel_checkpoint.py \
  reference_methods/INTEL/source/model/model.ckpt "$work/intel.npz"
CUDA_VISIBLE_DEVICES=1 python -u models/eval_reference_deadline.py intel \
  --checkpoint "$work/intel.npz" --output "$out/intel" > "$out/intel.log" 2>&1
python train_scripts/v11/prepare_ppo_lwd.py reference_methods/PPO-LwD/source "$work/lwd"
export PYTHONPATH="$work/lwd:$PWD/models${PYTHONPATH:+:$PYTHONPATH}" DGLBACKEND=pytorch
CUDA_VISIBLE_DEVICES=0 python -u train_scripts/v11/train_ppo_lwd_rb.py \
  "$rb_train" "$rb_validation" "$out/ppo_training/actor.pt" > "$out/ppo_training.log" 2>&1
CUDA_VISIBLE_DEVICES=1 python -u models/eval_reference_deadline.py ppo_lwd \
  --checkpoint "$out/ppo_training/actor.pt" --output "$out/ppo_lwd" > "$out/ppo_lwd.log" 2>&1
python tools/summarize_reference_deadline.py outputs/v11/reference_deadline_20261119 \
  --follow-up outputs/v11/reference_deadline_20261119_followup --learned outputs/v11/learned_deadline_20261119 \
  --supervised outputs/v11/supervised_deadline_20261119 --intel-ppo "$out" \
  --fused outputs/v11/fused_pave_02s_deadline --output "$out/summary.json"
```

`rb_train` 和 `rb_validation` 分别指向前述 LTFT 生成器的训练与独立验证目录。PPO 需要前述 GPU 版 DGL 环境。计分节点编号沿用外部基线的 `graph_nodes.jsonl`。 [INTEL 源码与权重](https://github.com/isl-org/NPHard) · [PPO-LwD 源码](https://github.com/sungsoo-ahn/learning_what_to_defer)

## 论文消融

完整的 CUDA Graph × 编译实验统一通过以下入口运行。脚本先完成固定 130 步，再启动 200 ms 截止实验。每阶段对四臂分别运行作者测试集的 1000 张图和三个种子，按种子调整四臂顺序。

```bash
out="outputs/v11/paper_ablations/cuda_graph_compile_$(date +%Y%m%d_%H%M%S)"
bash train_scripts/v11/paper_2x2.sh "$out" 1
python tools/summarize_paper_2x2.py "$out"
```

200 ms 解码时机消融配对比较 `full` 和 `terminal`。两臂均开启搜索捕获和策略编译，使用同一批图、种子、权重及 top 4 A.3 解码。

```bash
out="outputs/v11/paper_ablations/decode_schedule_$(date +%Y%m%d_%H%M%S)"
bash train_scripts/v11/paper_decode_schedule.sh "$out" 1
python tools/compare_mis_runs.py "$out/full" "$out/terminal" --output "$out/summary.json"
```

200 ms 解码宽度对照使用相同的 200 个粒子，在中段和末次分别解码能量最低的 4 个或全部 200 个。OnlineMIS 独立复测沿用 180 ms 内部搜索、200 ms 端到端截止，以种子 1023、1025、1027 配对重跑两种方法。

```bash
out="outputs/v11/paper_ablations/decode_width_$(date +%Y%m%d_%H%M%S)"
bash train_scripts/v11/paper_decode_schedule.sh "$out" 1 decode_all
python tools/summarize_decode_width.py "$out"
panel="outputs/v11/online_panel_$(date +%Y%m%d_%H%M%S)"
bash train_scripts/v11/paper_online_panel.sh "$panel" 1
python tools/summarize_online_panel.py "$panel"
```

权重与粒子数桥接对照通过同一融合入口运行作者发布的 checkpoint，分别取 20 和 200 粒子。报告沿用首次正式评测的本地 K=200 结果及作者 K=20 的 100 步封顶结果，按图和种子配对。

```bash
out="outputs/v11/paper_ablations/checkpoint_bridge_$(date +%Y%m%d_%H%M%S)"
bash train_scripts/v11/paper_checkpoint_bridge.sh "$out" 1
python tools/summarize_checkpoint_bridge.py "$out"
```

前导解码与同预留 PAVE 对照在相同作者图和种子上复用已存基线。`leading` 对解码宽度实验中同日运行的 `full`。`holdout` 关闭后处理、预留 40 ms，对主实验的 PAVE 开启与关闭两组。

```bash
out="outputs/v11/paper_ablations/decode_reserve_$(date +%Y%m%d_%H%M%S)"
bash train_scripts/v11/paper_decode_reserve.sh "$out" 1
python tools/summarize_decode_reserve.py "$out"
```

邻接表示消融保留 CUDA Graph，关闭两臂的策略编译。脚本先跑固定 130 步，再跑 200 ms 截止。两臂分别把 dense 和 CSR 归一化邻接交给冻结策略，搜索能量均用 CSR。

```bash
out="outputs/v11/paper_ablations/adjacency_$(date +%Y%m%d_%H%M%S)"
bash train_scripts/v11/paper_adjacency.sh "$out" 1 uncompiled
for mode in fixed130 deadline200; do
  python tools/compare_mis_runs.py "$out/$mode/dense" "$out/$mode/csr" \
    --output "$out/$mode/summary.json"
done
```

精度消融比较相同冻结权重的 FP16 和 FP32 求解配置。两臂均采用 dense 策略邻接、CUDA Graph 和策略编译，关闭 TF32，先跑固定 130 步，再跑 200 ms 截止。

```bash
out="outputs/v11/paper_ablations/precision_$(date +%Y%m%d_%H%M%S)"
bash train_scripts/v11/paper_precision.sh "$out" 1
for mode in fixed130 deadline200; do
  python tools/compare_mis_runs.py "$out/$mode/fp16" "$out/$mode/fp32" \
    --output "$out/$mode/summary.json"
done
```

单独运行各臂的方式如下。

```bash
for arm in full eager no_compile eager_no_compile; do
  FIXED_STEPS=130 RESULT_DIR="outputs/v11/paper_ablations/fixed130/$arm" \
    bash train_scripts/v11/paper_ablation.sh "$arm" 1
  RESULT_DIR="outputs/v11/paper_ablations/deadline200/$arm" \
    bash train_scripts/v11/paper_ablation.sh "$arm" 1
done
python tools/compare_mis_runs.py outputs/v11/paper_ablations/deadline200/full \
  outputs/v11/paper_ablations/deadline200/no_compile
```

所有运行默认采用 1000 图与三种子。上面的顺序可从头跑通两种协议。正式计时按种子轮换各臂的执行顺序。

| 配置 | 相对 `full` 的变化 |
| --- | --- |
| `full` | 捕获搜索、编译策略、dense FP16、top 4、中段与末次解码、关闭前导与 PAVE |
| `eager` / `no_compile` / `eager_no_compile` | 关闭搜索捕获、关闭策略编译、同时关闭两者 |
| `csr` / `csr_no_compile` / `fp32` | 策略邻接改为 CSR、CSR 且关闭策略编译、精度改为 FP32 |
| `decode_all` / `leading` / `terminal` | A.3 解码全部粒子、开启前导解码、关闭中段解码 |
| `pave` / `holdout` | 同预留 40 ms，运行 PAVE 或保留这段预算 |

`CHECKPOINT_DIR` 和 `NUM_K` 用于作者权重和粒子数桥接。`FIXED_STEPS` 由块长 10 向上对齐，`GRAPHS` 和显式种子可用于冒烟。`BUDGET` 以秒计，`PAVE_RESERVE_MS` 默认 40。 [正式消融方案](../../docs/论文实验.md#正式消融顺序)

## 0.2 秒融合限时对照

在 Linux 项目根目录运行。第一个参数是 GPU 编号，默认为 1。脚本按 PAVE 开关和三个种子先后启动单进程评测，每次对照使用新结果目录。

```bash
RESULT_DIR="outputs/v11/fused_pave_02s_$(date +%Y%m%d_%H%M%S)" \
  bash train_scripts/v11/fused_pave_02s.sh 1
```

需要作者测试集缓存 `models/data/MIS_tensors/RB_small_author_test`、v2 骨干 `outputs/v2/none_seed0/checkpoints` 和种子 0 的 PAVE-Stats H1。评测参数为 K=200、dense FP16、`torch.compile`、A.3 top 4，无前导解码，预算 0.2 s。每臂输出 `v11_rows.csv`、配置摘要和日志。 [结果与字段](../../outputs/README.md) · [项目现状](../../docs/项目现状.md)

早期融合和消融由 `v14_ab.sh`、`v14_ab01.sh`、`v14_fused.sh` 和 `v14_l1l2.sh` 复现。 [早期消融记录](../../docs/temporary/RLNN_v13_Ablation_Results.md)

## v11 展开训练与 FIFO 诊断

历史实验依据见 [v11 方向与判据](../../docs/temporary/RLNN_v11_Direction_Locked.md)。默认单卡、单进程，文中的 GPU 1 对应原实验部署。

## 执行顺序（判死就停）

| 步 | 命令 | 产物 | 判死条件 |
| --- | --- | --- | --- |
| 0 本地 | `python -m py_compile models/{mis_v11_unroll,mis_v11_gates,probe_unroll_grad,mis_v11_tabu,probe_tabu_walk,summarize_v11}.py` | — | 编译不过 |
| 0 本地 | `python models/mis_v11_gates.py selftest` | `outputs/v11/gates_selftest.json` | 9 项任一 false |
| 1 | `bash train_scripts/v11/probe_grad.sh 1` | `outputs/v11/probe_unroll_grad*.json` | 方向/非零梯度/相关性任一不过（exit 3） |
| 2 | `bash train_scripts/v11/smoke4.sh 1` | `outputs/v11/smoke/**` | engine 闸门任一 false，或 `--fixed_steps 810` 没钉住步数 |
| 3 | `bash train_scripts/v11/a0_gate0.sh 1` | `outputs/v11/gate0/**` | 主闸门 (i) 判死 → **不许开第 4 步** |
| 4 | `bash train_scripts/v11/c3.sh 1 unroll all` | `outputs/v11/a1_c3/**` | — |
| 4′ | `bash train_scripts/v11/c3.sh 1 erdoes all` | `outputs/v11/a1p_ctrl/**` | — |
| 5 | `python models/mis_v11_gates.py audit --a …/a1_c3/run.json --b …/a1p_ctrl/run.json` | `outputs/v11/gates_audit.json` | 除 `loss` 外还有字段不同 |
| 6 | `python models/summarize_v11.py --a A0/v11_rows.csv --b A1/v11_rows.csv --boot 5000` | 终端表格 | S1 判据（§2.7：Δ 与 pooled 95% CI + 三 seed 稳健） |
| C4 | `python -u models/probe_tabu_walk.py --graphs 200 --outer_rounds 8 --h 8,16,32,64,128 --budget 0` | `outputs/v11/tabu_walk_200/**` | ER=0（FIFO 从不命中）或 AR=0/8 → 停，不调 h |

**t5 已给出判决**：第 1 步在 **n=200** 判死（梯度无方向信息，见 `RLNN_v11_Implementation.md` §3.4），
第 4 步不该启动；C4 探针 ER=0.000 在 h=8…128 全部相同 ⇒ 判「惰性」，不调 h 重试。

第 1 步不过就**不许**写/跑第 4 步（direction §2.9 硬门）。
第 3 步的主闸门 (i) 不过就**不许**开 A1（§2.7）。时钟口径 (ii) 只报告，不进对照池。

## 门 0 的两个口径（不要混）

* **(i) 步数匹配（确定性，主闸门）**：`--fixed_steps 808` → `cutoff=inf`、
  `max_blocks=ceil(808/10)=81`、步数 810；PAVE 的 0.2 s 预留按墙钟另给，
  所以「搜索的时钟」换成「搜索的步数」，与机器速度无关。
  判据：解 `20.0685 ± 0.0005`、平均步数 ∈ [700, 860]。
* **(ii) 时钟（只报告）**：同一套 flags 走 1.0 s deadline。必须记实际步数、ms/步、
  load 秒；步数差 >10% 记「机器速度不同」，结果标 `steps=`，**不得**混入 A0 对照池。
  看到 ≈3.9 ms/步、≈244 步 = 跑成了 fp32+CSR（`--dense` 没生效或混精度掉了），判配置错。

## 训练预算（`TP`）

冻结训练循环对每张图做 `--num_tp` 次更新（默认 100）。200 步展开实测 ≈88 s/图，
1000 图 1 epoch ≈ 24 h，**装不进** direction §2.6 的 8.2 GPU·h。因此 `c3.sh` 默认
`TP=2`（≈3 GPU·h）：`TP=2 bash train_scripts/v11/c3.sh 1 unroll all`，
若显式用 `TP=100` 必须当作协议偏离记录在实验日志里。

## 与既有材料的关系

* `validation/k_sweep_v11.sh` / `validation/k_sweep_steps.py` 是 t4 的 200 图 K 扫描，
  **已关线**（K=200 最优，见 MATLAB 报告 I2），不要重跑。
* `models/eval_v8_padded.py --fixed_steps` 用块数控制 A0 的搜索，`--v11_csv` 另写逐图记录。v13 的限时路径另记解可用时刻与原始墙钟。 [A0 实现](../../docs/temporary/RLNN_v11_Implementation.md)
* 训练侧 loss 走 `--loss unroll`（`models/mis_v11_unroll.py`），冻结文件的接线点
  在 `docs/temporary/RLNN_v11_Implementation.md` 有「文件:行」清单。
* C4/L04 全部在 `models/mis_v11_tabu.py` + `models/probe_tabu_walk.py` 里，**没有**改
  `models/mis_v3_search.py`（冻结）：`repair_tabu` 补的正是冻结版 `:247 repair()` 缺的 tabu 参数。
