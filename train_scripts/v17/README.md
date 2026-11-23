# 四档限时对比

四个测试集共 2,144 张作者图。200 ms 的结果逐图记录种子 1017、1019、1021 交付的最大独立集、交付时点及求解器实际返回时间。主表和附录的同图定性比较共用这些记录。

## Gurobi

使用 Gurobi 13.0.3 的 Python 接口。每个顶点对应一个二元变量和一条邻域约束。逐图从主机边表开始计时，包含构造稀疏矩阵、建立模型、优化、复制可行解和释放模型。每次求解占用一个 CPU 核。多个核仅用于并行处理不同图，每个核载入独立的常驻求解环境。求解器回调交付的顶点集合在计时结束后逐边验证。

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python models/eval_gurobi_deadline.py \
  --dataset models/data/MIS_tensors/ER_small_author_test \
  --output outputs/v17/four_scale_comparators/ER_small/gurobi --cores 38 39
```

小 RB、大 RB、小 ER 的 `per_graph.csv` 和 `solutions.jsonl` 保存全部实测。万节点 ER 超过当前许可的 2,000 变量和 2,000 线性约束限制，对应的 `status.json` 记录首张实际模型的规模和错误码。报告中用单独的状态表示这一项。

## DIMES

四档共用在独立小 ER 图上本地训练的冻结权重。上游模型、训练规模、打分与解码对拍、四档推理命令见 [DIMES 实验说明](DIMES.md)。

## 图表

```bash
python tools/summarize_scale_deadline.py outputs/v11/scale_deadline_20261120 \
  --paper-csv outputs/v16/four_scale_200ms.csv --tex overleaf/sections/scale_table.tex
python tools/plot_cross_scale_witnesses.py
```

主表 CSV 的每个方法与图集组合均有来源路径。定性图固定每档的 `idx_0`，分别比较相同种子下的顶点成员，保留未按时交解的运行。
