# 实验输出

真实运行位于 `v000_rlnn_baseline/<run>/`。每个运行保留 run.json、日志和 metrics/metrics.csv；权重及图数据在服务器保留，不提交。smoke 由元数据 kind 标识，单独整理结果。运行状态与结果以各运行目录的 run.json、日志和 CSV 为准。
v1 的正式运行放在 outputs/v1/<run>/，逐图指标保存在 per_graph.csv，参数、耗时和峰值显存在 run.json。固定时间和全历史解码诊断使用各自的运行目录。见 [v1 实验协议](../train_scripts/v1/README.md)。

