# 逐图截止与常驻求解器

三个求解器从相同的主机端 `edge_index` 开始计时。构造 CSR、求解器内部图表示、学习版的 GPU 往返和把解复制到主机档案均占预算。编译、权重加载和暖机在实例计时之前。每个求解器有一个 CPU 线程，学习版另用一张 RTX 5090。求解结束后按原边表核验已归档的节点集。

## 作者测试图

RB-[200–300] 和 RB-[800–1200] 各 1,000 图，ER-[700–800] 共 128 图，ER-[9000–11000] 共 16 图。各图使用 1017、1019、1021 三个种子。对每个预算独立启动整组求解。每次保留 3 ms 返回时间。RB 两档的分支时间按原 200 ms 配置的比例缩放到整数毫秒，ER 只做交换。所有截止使用同一个已冻结检查点与打分权重。

下面的命令在仓库根目录执行。OnlineMIS 使用仓库内固定版本的 KaMIS 源码和其已配置的 CMake 构建目录。

```bash
python tools/build_resident_online_mis.py
export CUDA_VISIBLE_DEVICES=1
for split in RB_small RB_large ER_small ER_large; do
  regime=$split
  if [[ $split == ER_large ]]; then regime=ER_small; fi
  for ms in 50 100 200 500; do
    out="outputs/v16/anytime_author/$split/$ms"
    if [[ $split == ER_large && $ms == 200 ]]; then out=outputs/v16/er_large_200ms; fi
    for arm in learned degree online_mis; do
      taskset -c 11 bash train_scripts/v16/anytime.sh "$regime" "$arm" \
        "models/data/MIS_tensors/${split}_author_test" \
        "$out/$arm" "$ms" 1017 1019 1021
    done
  done
done
python tools/summarize_resident_deadlines.py
```

`summary.csv` 中的 `ready` 统计截止前已经复制到主机的合法解，`returned` 统计求解调用本身也在截止前完成的次数。每个图–种子对取截止时最大可行集，未交付记 0 分。`paired.csv` 先在每张图内平均三个种子的差，再以图为单位重采样 5,000 次。论文折线的每个点对应一次独立预算实验。

在 ER-[700–800] 上另做时间批次敏感性实验。同一进程绑定一个 CPU 核并持有三个已载入的求解器，每个图–种子块内运行全部 12 个方法–预算组合。384 个块按 Williams 平衡顺序轮转，确保每个组合在 12 个位置各出现 32 次。输入和打分规则与上表相同。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 \
  taskset -c 24 python train_scripts/v16/interleave_er.py
```

## DIMACS 寄存器干涉图

纳入 COLOR 目录中标为 REG 的全部 14 张图，保留所有顶点，沿用原边表示冲突。三个求解器均用 200 ms 预算。学习和度排序沿用相同的纯交换调度。数据不参与训练和参数选择。[COLOR 实例目录](https://mat.tepper.cmu.edu/COLOR/instances.html) · [DIMACS 原始数据](https://archive.dimacs.rutgers.edu/pub/challenge/graph/benchmarks/color/)

```bash
python train_scripts/v16/prepare_dimacs_register.py models/data/v16_dimacs_register
CPU_CORE=24 bash train_scripts/v16/dimacs_register.sh
python tools/summarize_dimacs_register.py models/data/v16_dimacs_register outputs/v16/dimacs_register
```
