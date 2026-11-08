# v3 合法平台搜索与收益排序

冻结原 RLNN，使用 K=20、T=100、FP16 和原 A.3。初始解做三轮 v2 rule23，之后最多八个外层轮。严格改善优先；停滞时用合法 1→1 动作改变节点阻挡关系，首动作后执行 H=1 或 H=2 的固定继续。每段最多四轮规则改善。根候选最多32个，其中覆盖度优先一半，其余按固定随机节点顺序选取；后续严格交换最多48个。

`mis_v3_search.py` 的 `rollout` 同时用于标签和部署。历史禁忌只过滤外层首动作，内部只禁止立即反向；达到宏动作深度即返回，规则补段也计入八轮上限。可选 anchored 2→1 固定新加入节点直到首段结束，终点若小于已保存最佳解则回退。

## 运行顺序

1. `starts.sh` 在指定 GPU 生成合法起点。`SEED` 为骨干种子，`STREAM` 为搜索随机流，`SPLIT` 为 train/validation/test，`COUNT` 控制图数。
2. `diagnose.sh` 对100张验证图、三个骨干种子的300个起点做 D0后备、D1联合、D2中性H1、D3中性H2普查，报告全候选及cap32的oracle差。
3. 验证出现排序空间后运行 `sweep.sh collect-train`。两个训练随机流与三个验证种子分别采集起始和中间状态，H1/H2独立生成标签、独立准入；通过后在两卡并行训练各三个统计头。
4. 在验证集比较完整平台求解的规则与统计头。统计头超过最强规则后，再训练普通消息/一阶频率编码，最后在1000张测试图上做正式比较。

```bash
export PYTHON=/home/hzeng/envs/rlnn/bin/python
CUDA_VISIBLE_DEVICES=0 SEED=0 SPLIT=validation COUNT=100 bash train_scripts/v3/starts.sh
bash train_scripts/v3/diagnose.sh
bash train_scripts/v3/sweep.sh collect-train
```

准入要求验证最终起点至少10张图有非平局、全状态oracle比最强规则高至少0.03，训练集至少100个非平局状态且覆盖50张图。结果写 `decision.json`，H1独立写 `decision_h1.json`。没有训练标签时只记录验证准入，不启动训练。

## 模型与评价

统计头为24→64→32→1。普通消息和频率头使用相同8维节点特征、两条8→32→32分支及相同动作池化。普通消息输入 F/MF，频率输入 (F+MF)/2 与其残差，参数量和邻居传播次数一致。训练按状态平衡，每状态最多64个非平局对，32状态/批，30轮。默认学习率统计头1e-3、节点编码3e-4。

`eval.sh` 支持 none、rule23、random、degree、coverage、pave。固定工作量下多个后处理共享同一起点；固定时间一次只评一个策略。总时间包含拓扑、搜索、解码、特征、排序、后续搜索和可行性检查，预载与原始H2D沿用v2口径不计入。`--extended-search` 允许原RLNN在截止前超过100步，作为充分使用预算的强对照。

```bash
SEED=0 CUDA_VISIBLE_DEVICES=0 bash train_scripts/v3/eval.sh \
  --save_dir outputs/v000_rlnn_baseline/rb_train_author_b64_g8_gpu1_seed0/checkpoints \
  --label RLNN-v3 --post none rule23 random degree coverage \
  --result_dir outputs/v3/test_rules_seed0
```

训练任务可同卡并行；正式延迟每张卡同时只运行一个任务，两卡运行不同组。普查和标签使用多核CPU。权重、完整标签和数据缓存仅保留服务器；`run.json`、逐图CSV、txt汇总同步到本地。`tools/summarize_v3.py` 与 `tools/update_v3_workbook.mjs` 更新工作簿的V3页。
