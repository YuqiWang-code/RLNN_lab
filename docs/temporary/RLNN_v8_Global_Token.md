# v8 全局 token：一个中性结果

> 2026-11-14。路线第 1 步（`README.md` 的「下一步路线」）：给 5 层 GCN 加图级上下文，半天成本，
> 判据是 1.0 s ≥ 20.00 才留下。结论：**没有可测的收益，判据未过，进第 2 步。**

## 0. 做了什么

`models/mis_v8_global.py` 的 `GlobalTokenGCN` 逐层照抄 `models/models.py:116 GCN.project_hidden`
（`x = activation(conv(x) + linear(x)) + x`），只在激活前多加一项图级广播：

```python
nei = conv(x, adj)
x_ = linear(x)
pooled = torch.cat((x.mean(dim=0), x.amax(dim=0)), dim=-1)   # 图级摘要
x = activation(nei + x_ + token(pooled).unsqueeze(0)) + x
```

`--arch {gcn,global}` 接在 `models/main.py:build_parser()`，`models/solvers.py` 的 `RLNN.__init__`
按 arch 选模型；warm start 允许缺 `tokens.*`（`load_state_dict(strict=False)`），其余键完整加载。
训练入口 `train_scripts/v8_macro/global_train.sh`（`ARCH`/`WARM_START`/`FINETUNE_LR` 可覆盖）。

## 1. 协议

- 从零训练（`EPOCHS=50`、1000 图、`num_tp=50`、`lr 1e-4`）**不可行**：第 6 个 epoch 验证目标
  只有 17.5，而冻结骨干的原始训练是**热启动**的（`outputs/v2/none_seed0/run.json` 里有
  `warm_start`，epoch 1 就到 19.52）。从零 200 图 1 epoch 的对照也说明这不是 arch 的问题：
  `gcn` 17.37、`global` 17.24，两者同档。
- 因此按部署口径做**微调**：`WARM_START=outputs/v2/none_seed0/checkpoints/best.pt`、
  `FINETUNE_LR=2e-5`、6 epochs、1000 图。验证目标 19.48 → 19.55 → 19.47 → 19.45 → 19.46 → 19.47
  （冻结基线 19.52–19.53）：**验证集上就是平的**，与之前 `ft_tp50` 微调的结论一致。
- 部署评测：`train_scripts/v8_macro/global_eval.sh`，500 张作者测试图、0.5/1.0 s、
  `captured/cached`、单种子 1017；两个检查点跑在同一批图上，用
  `models/compare_v8_runs.py` 做图级配对自助（5000 次）。

## 2. 结果（500 图，图级配对）

| 时钟 | 冻结骨干 | + 全局 token（微调 6 epoch） | 配对差 | 胜/平/负 |
| --- | --- | --- | --- | --- |
| 0.50 s | 19.9600 | 19.9420 | −0.0180 [−0.0480, +0.0120] | 25/441/34 |
| 1.00 s | 19.9900 | 19.9920 | +0.0020 [−0.0220, +0.0280] | 22/457/21 |

两个格子都跨零，1.0 s 的绝对值 19.9920 也没到判据要求的 20.00。

## 3. 判读

全局 token 在这条管线里**不提供可测增益**。原因大概率不在「感受野不够」：MIS 的能量是
`x^T(β/2 A + I)x` 形式的**局部可加**目标，翻转决策只依赖该点的度、邻居占用数和二阶邻居，
5 层 GCN 在 250 节点（平均度 38）的图上已经能把 2–3 跳覆盖到，图级摘要带来的是**更慢的收敛**
而不是更好的决策。验证目标不动也说明优化面没变。

按路线继续第 2 步（学习型交换头）。这一条与全局 token 正交：它改的是**动作空间**，
不是特征。

## 4. 复现

```bash
# 微调（热启动冻结骨干，允许新增的 tokens.* 随机初始化）
EPOCHS=6 GRAPHS=1000 ARCH=global RUN_NAME=global_ft6 \
  WARM_START="$PWD/outputs/v2/none_seed0/checkpoints/best.pt" FINETUNE_LR=2e-5 \
  bash train_scripts/v8_macro/global_train.sh

# 同一批图上的配对评测
GRAPHS=500 ARCH=gcn    RUN_NAME=none_seed0 RESULT_DIR=outputs/v8_macro/global_eval_base \
  bash train_scripts/v8_macro/global_eval.sh
GRAPHS=500 ARCH=global RUN_NAME=global_ft6  RESULT_DIR=outputs/v8_macro/global_eval_ft6 \
  bash train_scripts/v8_macro/global_eval.sh
python models/compare_v8_runs.py \
  --base outputs/v8_macro/global_eval_base/capture_budget/capture_budget_ref0.csv \
  --test outputs/v8_macro/global_eval_ft6/capture_budget/capture_budget_ref0.csv
```

## 5. 产物

- 训练：`outputs/v2/global_ft6/{checkpoints,metrics,train.log}`、`outputs/v8_macro/queue/global_global_ft6.log`
- 评测：`outputs/v8_macro/global_eval_base/capture_budget/capture_budget_ref0.csv`、
  `outputs/v8_macro/global_eval_ft6/capture_budget/capture_budget_ref0.csv`
- 对照（从零训练）：`outputs/v8_macro/queue/global_{global_smoke,global_tp50,ctl_gcn,ctl_global}.log`
