# v4 频率机制诊断

使用已训练权重分析频率编码、分支行为、动作收益和计算成本。PAVE-Stats H1、MIS能量、原训练损失和A.3解码保持原协议。

- 方案：`docs/temporary/RLNN_v4_Frequency_Mechanism_Diagnostic_Implementation_Plan.md`
- 输出：`outputs/v4_frequency/followup/`
- 训练图前20张用于机制探索，验证图100张用于复核；图顺序取已有tensor manifest，不重新生成图。
- P1使用FP32，K20/T100完整轨迹，观察前4粒子，分别记录冻结教师轨迹和模型自身轨迹。
- P2分别运行FP32/FP16；每图共享`x0`与逐步`U`，保留原按粒子最低能量档案。分支干预仍执行全部编码，运行时间不用于结构剪枝的速度结论。
- P3先用训练10图检查R2/4/8/16稳定性，再比较20图Uniform16与Stratified24；所有评分方法使用同一冻结教师的成对收益标签。只有发现值得复核的抽样定义才扩到验证100图。
- P4在独占GPU上交错测量方法，20图、30次预热、5次重复。端到端计时与组件同步微基准分开。

统计先在图内聚合，再计算每个训练种子的图均值；三种子标准差使用样本标准差。配对bootstrap以图ID为重采样单位，并保留该图所有训练种子。验证图已用于权重选择，诊断区间不代表独立测试结论。

环境与冻结输入记录：

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHON=/home/hzeng/envs/rlnn/bin/python
$PYTHON train_scripts/v4_frequency/followup/manifest.py --source-sha "$(git rev-parse HEAD)"
# 全部诊断结束后确认所有检查点保持原值
$PYTHON train_scripts/v4_frequency/followup/manifest.py --verify
```

正式1000图测试集不参与本轮诊断。后续是否短训由P1–P4形成的任务收益证据决定。

运行入口依次为 `smoke.sh`、`explore.sh`、`spectrum_confirm.sh`、`utility_controls.sh`、`branch_confirm.sh` 和 `profile.sh`。P2与P1可并行。P4在GPU 0的FP32回放结束后启动。结果用 `python tools/summarize_frequency_followup.py --plots` 聚合，用 `tools/update_frequency_followup_workbook.mjs` 更新追诊页。

完整逐层频谱、逐状态回放、逐动作重复收益保存在服务器的 `outputs/v4_frequency/followup/`。Git保留运行清单、图级求解结果、汇总表和图。
