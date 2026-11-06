# TSP100 基线

专用环境：`/home/hzeng/envs/nexco`，Python 3.10.19、PyTorch 2.9.1+cu130、Lightning 1.9.5。环境配置为 `models/environment-5090.yml`。脚本默认使用 GPU 1，运行目录名包含时间。设置 `RUN_NAME` 可指定独立的实验名称。

## 数据

仓库已保存本版训练和验证数据。重新生成时，在项目 `models/` 目录使用不同输出路径执行：

```bash
python generate_tsp_data.py --output data/tsp_train/tsp100_train_lkh100_10000_regenerated.txt --samples 10000 --seed 20261106 --workers 8
python generate_tsp_data.py --output data/tsp/tsp100_val_lkh100_128_regenerated.txt --samples 128 --seed 20261107 --workers 4
```

坐标独立均匀采样自 `[0,1]^2`，100 个节点。参考路径由服务器 `/usr/local/bin/LKH`（LKH 3.0.14）生成，RUNS=1、MAX_TRIALS=100、坐标放大至 1,000,000 后按 EUC_2D 求解。种子由数据划分种子和实例序号确定。每个数据文件旁保存生成参数 JSON。训练集与验证集独立生成，测试集沿用仓库的 `tsp100_concorde_7.75585.txt` 共 1,280 个实例。

## 训练和测试

在项目根目录执行：

```bash
bash train_scripts/v001_tsp100/train_tsp100.sh
CKPT_PATH=/absolute/path/to/trained/checkpoint.ckpt bash train_scripts/v001_tsp100/test_tsp100.sh
```

训练使用 12 层、隐藏维度 256 的 GNN，从零初始化，seed=1234，batch size=12，FP32，1 epoch。每轮实际使用 9,996 个训练实例，最后 4 个实例由 `drop_last=True` 丢弃，共 833 个优化步骤。优化器为 AdamW，初始学习率 0.001，weight decay=0.0001，余弦衰减。一致性参数 alpha=0.5。

验证使用全部 128 个独立实例，rewrite_steps=1、rewrite_iterations=1，关闭 2-opt。按 `val/solved_cost` 保存最优检查点，同时保存 last。

测试使用训练所得检查点、单卡、单次并行与顺序采样，rewrite_steps=5、rewrite_iterations=5、rewrite_ratio=0.1、m1=0.9、m2=0.7，2-opt 上限 5,000 次。记录原始解与扩展后解的平均目标值、逐实例 gap 均值、逐实例耗时和显存。gap 为相对对应实例 Concorde 参考路径的百分数。

## 实验范围

本版是一轮训练的流程基线。训练标签来自 LKH 启发式求解，测试标签来自 Concorde。它为后续扩大训练集与增加轮数提供固定入口，完整论文复现需要单独指定训练规模与预算。
