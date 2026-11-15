# v8 学习式退火：在接受准则上做任何改动都没有余地

> 2026-11-15。对应 README `## 下一步路线` 的第 3 步。判决：**在训练价值头之前就停掉**，
> 因为「接受准则」不是瓶颈——把策略软化只会更差，退火只能把损失收回来一部分。

## 0. 问题

前两步（全局 token、学习型交换头）都在训练前被判死或测成中性，剩下的假设是：
部署策略太贪心，卡在 19.98 是因为不敢走「上坡」，所以改接受准则（温度/退火/
学出来的 Metropolis）能换到另一个盆地。这个假设的依据是论文自己：纯 SA 的 RLSA
（19.97 @ 0.070 s）比他们自己的神经网络版 RLNN（19.52 @ 0.197 s）强 0.45 且快 2.8 倍。

判决实验不需要任何训练：把冻结策略的 logits 除以温度 `T` 再 sigmoid
（`p = sigmoid(logit / T)`），就得到一个「愿意翻转它本来不愿翻的点」的策略。
如果**任何**固定温度都打不过 `T = 1`，那么以温度为输入的学习式接受准则
（只能在这些温度之间或之上做插值）也不可能赢——先测量，再决定要不要训。

## 1. 实现

`models/mis_v8_pad.py` 的捕获块里加了一个设备标量 `self.temperature`：

```python
logits = self.model(self.state.unsqueeze(-1), None, self.normalized).squeeze(-1)
probs = torch.div(logits, self.temperature).sigmoid()
noise = torch.rand(self.state.shape, ...)
flipped = torch.where(noise < probs, 1 - self.state, self.state)
```

它在 replay 时被读，所以换温度**不用重建 CUDA 图**。`run(..., temperature=T, anneal=False)`：

- `anneal=False`：整段预算用同一个 `T`；
- `anneal=True`：`T(t) = 1 + (T0 - 1) * 剩余时间 / 总时间`，线性退到 1（块间由 host 更新）。

`models/eval_v8_padded.py` 加 `--temperatures` / `--anneal`，每格一个温度，cell 名
`padded/cached/T4`（退火加后缀 `a`），`T = 1` 时保持旧名字 `padded/cached` 以免旧 CSV 失配。
启动器 `train_scripts/v8_macro/temperature.sh`（单进程：四分片并发会让 kernel 发射串行化，
步数不可比）。汇总 `models/summarize_v8_temperature.py`。

## 2. 固定温度（100 张作者测试图，seed 1017，摊薄引擎）

| 时钟 | T=1 | T=2 | T=4 | T=8 |
| 0.20 s | **19.7800** | 19.7000 (−0.080 [−0.160,+0.000] 5/82/13) | 19.4200 (−0.360 [−0.460,−0.260]) | 19.0800 (−0.700 [−0.850,−0.560]) |
| 1.00 s | **19.8900** | 19.9000 (+0.010 [−0.050,+0.070] 5/91/4) | 19.5200 (−0.370 [−0.480,−0.260]) | 19.2100 (−0.680 [−0.820,−0.540]) |

单调下降：温度越高越差，`T=4` 已经掉 0.37，`T=8` 掉 0.68。唯一「不亏」的格子是
1.0 秒的 `T=2`（+0.01，CI 跨零，91/100 图完全相同）。

## 3. 退火（T0 起高、线性降到 1）

| 时钟 | T=1 | 2→1 | 4→1 | 8→1 |
| 0.20 s | **19.7800** | 19.7400 (−0.040 [−0.120,+0.050] 7/82/11) | 19.7100 (−0.070 [−0.150,+0.010]) | 19.6100 (−0.170 [−0.280,−0.060]) |
| 1.00 s | **19.8900** | 19.9000 (+0.010 [−0.060,+0.070] 6/89/5) | 19.8600 (−0.030 [−0.100,+0.030]) | 19.8000 (−0.090 [−0.170,−0.010]) |

退火确实把恒定高温的损失收回了大半（1.0 秒 `T=8`：−0.68 → −0.09），但**没有一个格子
显著为正**：最好的仍是 2→1 的 +0.01，CI 跨零，89/100 图打平。

## 4. 判读

1. **接受准则不是瓶颈。** 软化策略只能降低解的质量；退火把它救回来的量恰好等于
   「早期多点随机性」的收益，也就是 0。
2. 这与 `RLNN_v8_Exchange_Headroom.md` 是同一件事的两面：解除收敛解时
   1-flip、(1,1)、(1,2)、(2,3)、k≤4 的 k→k+1 **全是零增益**，所以「更愿意走上坡」
   也换不到新盆地——上坡走到的每个点仍在同一个 1-flip 闭包里。
3. 学习式退火（价值头 + 学出来的接受准则）只是在温度空间里做插值/条件化，
   而温度空间的**最好点就是基线本身**。没有可学的信号，训练只会学到 `T = 1`。
   因此第 3 步同样**在训练前判死**，不投入价值头的数据采集与训练。
4. 剩下唯一没被证伪的方向是**换移动集**而不是换接受准则：一次移动 ≥5 个点
   （KaMIS 的缩减/重启那一类），或者接受「解的质量天花板就是 19.98」。

## 5. 复现

```bash
# 固定温度
GRAPHS=100 BUDGETS="0.2 1.0" TEMPERATURES="1 2 4 8" \
  RESULT_DIR=outputs/v8_macro/temperature bash train_scripts/v8_macro/temperature.sh
# 退火
ANNEAL=1 GRAPHS=100 BUDGETS="0.2 1.0" TEMPERATURES="1 2 4 8" \
  RESULT_DIR=outputs/v8_macro/temperature_anneal bash train_scripts/v8_macro/temperature.sh
# 汇总
python models/summarize_v8_temperature.py \
  outputs/v8_macro/temperature/padded/padded_ref0.csv --label "constant T"
python models/summarize_v8_temperature.py \
  outputs/v8_macro/temperature_anneal/padded/padded_ref0.csv --label "annealed"
```

## 6. 产物

- `outputs/v8_macro/temperature/padded/padded_ref0.{csv,json}`（固定温度，800 格）
- `outputs/v8_macro/temperature_anneal/padded/padded_ref0.{csv,json}`（退火，800 格）
- `outputs/v8_macro/temperature_smoke/`（3 图冒烟）
- `outputs/v8_macro/queue/temperature{,_anneal}.log`
