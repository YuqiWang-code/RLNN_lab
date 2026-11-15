# models/closed — 已关闭方法线的代码

这些文件不属于现行流程，也没有被 `models/` 下的活跃模块或 `train_scripts/` 引用。留在这里是为了让归档文档（`docs/已归档的想法和文件/`）里的路径仍然可读。

**它们不能原样运行**：同目录移动后，`from eval_v2 import ...` 这类平铺 import 会失效。要复跑某条线，先 `git show <commit>:models/<file>.py` 拿回根目录。

| 文件 | 属于 |
| --- | --- |
| `eval_structure.py`, `train_structure.py`, `mis_structure.py`（仍在根目录，`solvers.py` 引用） | 结构补偿线（v1/v2，No-go） |
| `eval_static_prior*.py`, `static_prior_protocol.py`, `diagnose_static_leverage.py`, `train_static_direction.py` | static prior 线（No-go） |
| `eval_v7_context.py`, `diagnose_v7_context.py`, `profile_v7_context.py`, `train_v7_context.py`, `check_v7.py`, `eval_v7_tccr.py` | v7 context / TCCR 线（budget_fail，已归档） |
| `train_exchange_value*.py`, `train_flip_value.py`, `probe_direction_agreement.py` | v2/v3 价值头与方向一致性探针 |
| `eval_author_rb.py` | 早期作者集单点评测（被 `eval_v8_capture_budget.py` 取代） |