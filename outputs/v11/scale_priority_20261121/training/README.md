# Fused RLNN 节点排序训练

两次训练都从融合求解器使用的五层、128 通道 RLNN GCN 权重开始，在同一组独立生成的图上微调整个编码器。三档共用一份权重。选模指标是逐图静态贪心 SIZE 相对静态度排序的差，除以该图最好的教师 SIZE，先在每档内平均，再对三档等权平均。同分时取验证损失较小的一轮。[初始权重](../../../v2/none_seed0/checkpoints/best.pt) · [网络实现](../../../../models/scale_priority.py)

| 独立验证的单次贪心排序 | RB-[200–300] | RB-[800–1200] | ER-[700–800] |
|---|---:|---:|---:|
| 静态度升序 | 16.4375 | 33.6250 | 34.3750 |
| RLNN 优先级，初始权重 | 15.3750 | 31.3750 | 29.8750 |
| RLNN 优先级，选中第 1 轮 | **16.5625** | 32.8125 | 33.6250 |
| 静态度加神经残差，选中第 0 轮 | 16.4375 | **33.6250** | **34.3750** |
| 静态度加神经残差，仅在已训练轮次中取最优，第 20 轮 | **16.5625** | 33.5000 | 33.4375 |

静态贪心沿给定排序依次选取兼容节点，同分时保留节点编号顺序。正式 200 ms 求解器另有动态最小剩余度初始化、局部搜索和精化阶段，其计分结果由逐图时间戳确定。

第一套排序器从节点度生成连续初始占用值，复用 GCN 的五层残差消息传递和原输出层，再由轻量读出融合隐藏态与结构特征。24 个 epoch 中，第 1 轮获选。原编码器 165,505 个参数全部发生更新，参数变化的 L2 范数为初值的 1.715%。冻结模型经部署接口重载，在全部 48 张验证图上重现选模 SIZE。[冻结权重](best.pt) · [逐图验证](frozen_validation.csv) · [参数变化](frozen_validation.json) · [逐轮记录](epochs.csv)

第二套将负标准化度直接作为节点分数的起点，并用同图内度数相差至多 2 的节点对监督残差读出。配对两端的教师入选频率不同。残差末层以零初始化，幅度限制为 0.35。第 0 轮的排序精确复现静态度，在预定指标上获选。这套读出未进入正式候选。[第二套配置](../training_degree_residual/config.json) · [选模记录](../training_degree_residual/selection.json) · [逐轮结果](../training_degree_residual/epochs.csv)

## 图和监督

每档生成 128 张训练图和 16 张验证图，共 432 张。图生成种子为训练基值 4,100,000 或验证基值 4,200,000，加分布偏移和从零开始的图编号。RB-small、RB-large、ER-small 的偏移分别为 0、10,000、20,000。两档 RB 使用仓库的 Xu 生成器，并按节点数接受实例。ER 使用独立边概率 0.15，节点数在 700 到 800 间均匀采样。每张图在导出前独立重排节点编号。[生成脚本](../../../../train_scripts/v11/prepare_scale_priority.py)

KaMIS OnlineMIS 对每张图使用种子 283、977、1297、1601、2027、2741、3089、3371，各运行 0.5 秒。训练目标为八个合法解中最大的四个解的节点入选频率，大小并列时按上述种子顺序选取。归档包含全部 432 张图、3,456 个可行掩码、图的生成种子和逐解 SIZE。使用归档可复用同一组墙钟教师标签。[独立图与教师证书](labeled_data.tar.gz)

## 复跑

在仓库根目录及原项目 PyTorch 环境执行。原训练使用 RTX 5090、PyTorch 2.9.1+cu130、训练种子 8707，两套均为 24 个 epoch。下面的命令将结果写入独立目录。[训练入口](../../../../train_scripts/v11/train_scale_priority.py) · [优化器参数](config.json)

```bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
work=outputs/v11/scale_priority_reproduction
mkdir -p "$work/data"
tar -xzf outputs/v11/scale_priority_20261121/training/labeled_data.tar.gz -C "$work/data"
data="$work/data/scale_priority_20261121"
base=outputs/v2/none_seed0/checkpoints/best.pt
python -u train_scripts/v11/train_scale_priority.py --data "$data" --fused "$base" \
  --output "$work/fused_readout" --variant fused_readout --epochs 24 --seed 8707
python -u train_scripts/v11/train_scale_priority.py --data "$data" --fused "$base" \
  --output "$work/degree_residual" --variant degree_residual --epochs 24 --seed 8707
```

从图生成与教师采样开始重建数据时，可使用原实验服务器上的 OnlineMIS 可执行文件。

```bash
python -u train_scripts/v11/prepare_scale_priority.py \
  --output "$work/new_data" --binary tmp/kamis_build/online_mis \
  --train 128 --validation 16 --workers 8 --teacher-seconds 0.5
```
