# v3 合法平台搜索与收益排序

冻结 RLNN，使用 K=20、T=100、FP16 和原 A.3。初始合法解先做三轮 v2 rule23，之后最多八个外层轮。严格改善优先；停滞时通过合法 1→1 动作改变阻挡关系，再执行 H=1 或 H=2 的固定继续。每段最多四次规则交换，根候选最多32个，严格改善候选最多48个。

`models/mis_v3_search.py` 的 `rollout` 同时生成标签并执行部署动作。历史禁忌过滤外层首动作，内部禁止立即反向；规则补段也计入八轮上限。可选 `--anchored` 扩展到2→1，首段固定新加入节点，失败分支恢复最佳合法解。正式矩阵使用1→1。

## 环境与起点

服务器需已有原 RLNN 的三个种子检查点、`outputs/v2/none_seed{0,1,2}/checkpoints` 微调检查点，以及 `models/data/MIS_tensors/` 中的训练、验证和作者测试缓存。原骨干路径由 `common.sh` 的 `base_checkpoint` 指定。所有命令从项目根目录执行。

```bash
export PYTHON=/home/hzeng/envs/rlnn/bin/python
mkdir -p outputs/v3/queue
```

`starts.sh` 使用 `SEED` 指定骨干种子，`STREAM` 指定搜索随机流，`SPLIT` 取 train/validation/test，`COUNT` 指定图数。`CHECKPOINT` 和 `NAME` 可指定骨干目录及输出名。生成原骨干的300个验证起点、两个训练随机流，以及微调骨干的300个验证起点：

```bash
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$((seed % 2)) SEED=$seed SPLIT=validation COUNT=100 \
    bash train_scripts/legacy/v3/starts.sh > "outputs/v3/queue/starts_validation_seed${seed}.log" 2>&1 &
done
wait

for stream in 0 1; do
  CUDA_VISIBLE_DEVICES=$stream SEED=0 STREAM=$stream SPLIT=train COUNT=1000 \
    bash train_scripts/legacy/v3/starts.sh > "outputs/v3/queue/starts_train_stream${stream}.log" 2>&1 &
done
wait

for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$((seed % 2)) SEED=$seed SPLIT=validation COUNT=100 \
    CHECKPOINT="outputs/v2/none_seed${seed}/checkpoints" NAME="starts_tuned_seed${seed}" \
    bash train_scripts/legacy/v3/starts.sh > "outputs/v3/queue/starts_tuned_seed${seed}.log" 2>&1 &
done
wait
```

起点缓存用于质量选择和训练数据生成。正式端到端测试直接运行 RLNN，不将缓存读取时间替代搜索时间。

## 普查、准入与统计头

`diagnose.sh` 对同一批100张验证图、三个骨干种子的300个起点普查 D0后备邻域、D1联合邻域、D2中性H1、D3中性H2，分别保存完整候选和cap32候选的收益。

```bash
WORKERS=24 bash train_scripts/legacy/v3/diagnose.sh

"$PYTHON" train_scripts/legacy/v3/decide.py --validation outputs/v3/census \
  --variant D2_neutral_h1 --output outputs/v3/decision_h1.json
"$PYTHON" train_scripts/legacy/v3/decide.py --validation outputs/v3/census \
  --variant D3_neutral_h2 --output outputs/v3/decision.json
```

验证准入要求完整300个起点、100张图，至少10张图有非平局候选，全状态oracle比验证最强规则高至少0.03。没有训练标签时，决策文件中的 `train` 为null，`validation_pass` 表示是否值得生成对应标签。

H1/H2均通过验证准入后，批量生成独立标签并训练各三个统计头：

```bash
bash train_scripts/legacy/v3/sweep.sh collect-train
```

该入口同时采集两个训练随机流与三个验证种子的起始、中间状态，分别生成 `labels_train_h1/h2` 和 `labels_validation_h1/h2`；训练标签还需满足至少100个非平局状态、覆盖50张不同训练图。只有对应 `decision_h1.json` 或 `decision.json` 的 `train=true` 才启动训练。标签收集使用多核CPU，六个统计头分配到两张GPU并行运行。

单个已准入训练任务的接口为 `TRAIN_DATA VALID_DATA MODE SEED NAME GPU`：

```bash
DECISION=outputs/v3/decision_h1.json bash train_scripts/legacy/v3/train.sh \
  outputs/v3/labels_train_h1/states.pt outputs/v3/labels_validation_h1/states.pt \
  stats 0 stats_h1_seed0 0
```

训练按状态等权，每状态最多64个非平局对，32状态/批、30轮。三个头种子共享标签，只改变初始化和批顺序。默认学习率统计头1e-3、节点编码3e-4；检查点根据全部验证状态的实际动作收益选择，零候选、单候选和全平局状态均保留在评价分母中。

## 完整平台验证与节点编码

统计头训练完成后，在缓存的原骨干、微调骨干起点上比较完整求解结果：

```bash
"$PYTHON" train_scripts/legacy/v3/validate_heads.py --workers 24
```

输出 `validation_platform/` 和 `model_decision.json`。验证覆盖无后处理、rule23、random/degree/coverage的H1/H2，以及已存在的学习头。当前主比较采用微调骨干、H1统计头，以H2随机规则平台搜索为强规则参照。缓存并行验证用于选择质量，正式延迟由测试入口测量。

`train_encoders.sh` 读取 `model_decision.json`，仅在统计头超过最强验证规则时训练节点编码。当前选择H1，分别训练message/frequency的三个种子：

```bash
bash train_scripts/legacy/v3/train_encoders.sh
```

| 表示开关 | 结构 | 可训练参数 |
| --- | --- | ---: |
| `stats` | 24→64→32→1，SiLU | 3,713 |
| `message` | 两条8→32→32分支，输入F与MF；动作池化后评分 | 20,737 |
| `frequency` | 相同分支，输入L=(F+MF)/2与H=F−L；相同动作池化 | 20,737 |

节点输入、参数量、传播次数、候选池和监督标签在message/frequency之间一致。对移除、加入节点分别mean池化，对受影响邻域mean/max池化，再拼接24维动作统计。训练和验证使用32状态的分块对角图合批；部署在CPU一次评分整个候选池。

## 一轮策略分布适配与初始化对照

H1统计头有完整平台收益后，运行一次数据分布适配：

```bash
bash train_scripts/legacy/v3/adapt.sh
```

该入口复用原训练起点 `starts_train_seed0_stream0/1`，用 `stats_h1_seed0` 选择外层动作，标签继续策略保持固定。随后按origin分层混合约一半原规则状态、一半PAVE访问状态，总状态数和各origin数量保持原训练集规模。新标签重新准入后，训练三个 `stats_adapted_h1_seedN`；结构仍是3,713参数的统计头。

生成三个未训练统计头作为固定工作量对照，并更新完整验证：

```bash
"$PYTHON" train_scripts/legacy/v3/untrained_control.py
"$PYTHON" train_scripts/legacy/v3/validate_heads.py --workers 24
```

未训练头采用与训练器相同的seed0/1/2初始化，保存epoch0权重；它只进入微调骨干H1的固定工作量测试。

三个机制开关分别对应：

| 开关 | 设置位置 | 比较方法 |
| --- | --- | --- |
| 统计/普通消息/一阶频率 | `train.sh` 的MODE；评测从检查点读取mode | `stats_h1`、`message_h1`、`frequency_h1` |
| 继续深度H | 收集与评测的 `--horizon 1/2`，各自标签训练 | `stats_h1`与`stats_h2`及同深度规则 |
| 策略分布适配 | `adapt.sh` 使用混合训练状态 | `stats_h1`与`stats_adapted_h1`，同结构、同验证集 |

## 两张GPU的正式测试

节点编码与适配结束后，完整双卡入口为：

```bash
bash train_scripts/legacy/v3/run_tests.sh
```

入口等待 `queue/encoders.done.txt` 与 `queue/adapted.done.txt`，重做完整验证，再在100张验证图上比较20/50/100ms后处理预留，保存统一的 `budget_selection.json`。随后GPU0/1分别运行各自的fixed分片，再运行budget分片。

训练、起点生成可同卡并行；**正式延迟每张GPU同时仅一个评测进程**，两卡运行不同组。`test_matrix.py` 自动设置各任务的GPU和单线程数学库；服务器有 `taskset` 时，GPU0使用CPU核0–23，GPU1使用24–47。

固定工作量矩阵含原骨干H1、微调骨干H2、微调骨干H1，各三个种子。同一组内的后处理共享一次K20/T100搜索；微调H1还包含普通消息、频率、适配和未训练统计头。

0.5秒矩阵中，平台方法保留T100上限并使用验证选定的统一预留。原RLNN、微调RLNN的 `--extended-search` 强对照允许超过100步直到截止；微调RLNN＋rule23只预留20ms，其余时间用于搜索。每次预算评测只运行一个后处理策略。

总求解时间包含拓扑、搜索、A.3、特征构造、评分、后续搜索和可行性检查；数据预载和初始H2D沿用v2口径不计入。

### 单独继续fixed

```bash
"$PYTHON" train_scripts/legacy/v3/test_matrix.py --stage fixed --gpu 0 \
  > outputs/v3/queue/fixed_gpu0.log 2>&1 &
p0=$!
"$PYTHON" train_scripts/legacy/v3/test_matrix.py --stage fixed --gpu 1 \
  > outputs/v3/queue/fixed_gpu1.log 2>&1 &
p1=$!
wait "$p0"
wait "$p1"
```

### 单独继续budget

使用已保存的 `budget_selection.json`。需要单独完成验证预留选择时，执行：

```bash
bash train_scripts/legacy/v3/reserve.sh pave 0 &
p0=$!
bash train_scripts/legacy/v3/reserve.sh random 1 &
p1=$!
wait "$p0"
wait "$p1"
"$PYTHON" train_scripts/legacy/v3/select_reserve.py
```

运行或继续预算矩阵：

```bash
"$PYTHON" train_scripts/legacy/v3/test_matrix.py --stage budget --gpu 0 \
  > outputs/v3/queue/budget_gpu0.log 2>&1 &
p0=$!
"$PYTHON" train_scripts/legacy/v3/test_matrix.py --stage budget --gpu 1 \
  > outputs/v3/queue/budget_gpu1.log 2>&1 &
p1=$!
wait "$p0"
wait "$p1"
```

续跑以整组为单位：所有预期post均有 `status=completed`、`samples=1000` 的 `run.json` 时跳过该组；未完成组从头运行。各分片完成标记为 `queue/fixed_gpuN.done.txt` 或 `queue/budget_gpuN.done.txt`，完整入口结束后写 `queue/v3.done.txt`。

## 输出与汇总

全部产物位于 `outputs/v3/`。

| 目录或文件 | 内容 |
| --- | --- |
| `starts_validation_seedN_stream0/`、`starts_tuned_seedN/` | 原/微调骨干验证起点 |
| `starts_train_seed0_stream{0,1}/` | 两个训练随机流的起点 |
| `census/` | `per_state.csv`、`per_action.csv`、`states.pt`及普查汇总 |
| `labels_train_hH/`、`labels_validation_hH/` | H1/H2独立的完整标签 |
| `labels_train_pave_h1/`、`labels_train_mixed_h1/` | 策略访问状态与固定规模混合标签 |
| `{stats,message,frequency}_hH_seedN/` | 检查点、训练曲线、全部验证状态指标 |
| `stats_adapted_h1_seedN/`、`stats_untrained_h1_seedN/` | 适配与初始化对照 |
| `decision*.json`、`model_decision.json`、`budget_selection.json` | 标签准入、模型选择、时间预留选择 |
| `test_{original,tuned}_hH_seedN/POST/` | 固定工作量逐图结果 |
| `budget_VARIANT_seedN/POST/` | 固定时间逐图结果 |
| `queue/` | 各阶段日志与完成标记 |

逐图CSV保存图ID、骨干/头/搜索种子、base/rule/final Size、总时间、搜索/后处理/评分时间、动作数、候选数、截止信息和最终节点集合。`none/per_graph.csv` 的节点集合就是该fixed组共享的RLNN初始解。主表使用Mean Size的三个种子均值及样本标准差，同时报告相对RLNN和强规则的配对Δ、可行率、总参数与可训练参数、峰值显存。

```bash
"$PYTHON" tools/summarize_v3.py
```

该命令生成 `summary.json` 和 `summary.txt`。同步小型结果后，在具备项目artifact-tool运行环境的本地执行：

```bash
node tools/update_v3_workbook.mjs
```

工作簿 `docs/实验结果.xlsx` 更新 `V3汇总`、`V3逐种子`、`V3机制`。权重、图数据、完整起点与大规模标签留服务器；代码、配置、JSON/CSV诊断、txt快照和工作簿同步本地与GitHub。
