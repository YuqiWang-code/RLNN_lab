# v000_baseline

从服务器项目根目录执行 `bash train_scripts/v000_baseline/<脚本>.sh`。脚本先进入 `models/`，输出写入 `outputs/v000_baseline/<运行名>/`。

`train_*.sh` 为训练入口，`test_*.sh` 为预训练检查点评测入口。`train_er.sh` 与 `train_miser.sh` 均覆盖 ER 图训练。`test_misrb_large.sh` 评测 RB800–1200。

默认使用 GPU 0，设置 `CUDA_VISIBLE_DEVICES` 选择设备，设置 `RUN_NAME` 指定独立运行名。命令后追加的训练参数优先，例如：

```bash
RUN_NAME=tsp100_short CUDA_VISIBLE_DEVICES=1 bash train_scripts/v000_baseline/train_tsp100.sh --num_epochs 1 --validation_examples 16
```

依赖与数据准备状态见 `docs/项目现状.md`。新迭代在 `train_scripts/` 下建立新版本目录，同时在 `outputs/` 中使用对应版本名。
