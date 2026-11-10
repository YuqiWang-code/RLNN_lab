# 参考实现

[方法清单](catalog.tsv)给出作者仓库、论文和 v1 用途。每种方法的上游文件放在自己的 source 子目录，upstream.json 记录对应版本。RLSA 与 RLNN 使用同一上游项目。

~~~powershell
python tools/fetch_references.py all
~~~

下载只取公开的源码和论文。源码快照保存在本地与实验服务器，Git 跟踪清单、版本和本项目的适配入口。下载器过滤上游打包的权重、数据缓存和编译产物。已公开的部分实现按原仓库归档，IDG 的公开版本是 README 占位。

Mamba 主线参考了 [Mamba](Mamba/upstream.json)（Apache-2.0）、[Graph-Mamba](Graph-Mamba/upstream.json)（仓库根目录未提供许可证）、[NeuralWalker](NeuralWalker/upstream.json)（BSD-3-Clause）和 [Range-Measure](Range-Measure/upstream.json)（MIT）的作者代码。对应论文和用于 MIS 的定量对照见[文献索引](../docs/参考文献/文献索引.md)。

正式对照的训练和评估入口见 [v1 实验](../train_scripts/legacy/v1/README.md)。论文 PDF 按研究方向收在[参考文献](../docs/参考文献/文献索引.md)。

## v4 频域方法快照

16篇参考论文对应源码均已归档。`source/` 保留作者文件，`upstream.json` 固定版本；这些参考快照不等于在 RLNN 上完成复现。关键实现阅读见[方法与源码](../docs/已归档的想法和文件/v4_关键方法与源码.md)。

| 方法 | 本地源码 | 主要用途 | 快照许可证 |
| --- | --- | --- | --- |
| WaveNet | [source](WaveNet/source) | 图小波与非平稳信号 | 未提供根许可证 |
| PolyGCL | [source](PolyGCL/source) | 学习低/高通多项式视图 | 未提供根许可证 |
| SLOG | [source](SLOG/source) | 局部谱与归纳滤波 | MIT |
| S²GNN | [source](S2GNN/source) | 空间与频谱联合传播 | MIT |
| WaveGC | [source](WaveGC/source) | 奇偶 Chebyshev、小波尺度 | 未提供根许可证 |
| Spectral-benchmark | [source](Spectral-benchmark/source) | 输出频率与跨频学习诊断 | 未提供根许可证 |
| GRACE | [source](GRACE/source) | 低通/差分专家和节点门控 | 未提供根许可证 |
| HeroFilter | [source](HeroFilter/source) | 谱相关邻域 patch 与 Mixer | 未提供根许可证 |
| MORGAN | [source](MORGAN/source) | 谱频段专家与门控 | MIT |
| COPT-MT | [source](COPT-MT/source) | GCON、多任务组合优化迁移 | 未提供根许可证 |
| FreqFusion | [source](FreqFusion/source) | 自适应分频融合 | 子项目附许可证，未提供统一根许可证 |
| FADC | [source](FADC/source) | 局部频率选择与感受野 | Apache-2.0 |
| FDConv | [source](FDConv/source) | 固定参数预算下的频域动态卷积 | MIT |
| F2Net | [source](F2Net/source) | 跨频融合与梯度平衡 | MIT |
| GCON | [source](GCON/source) | 多滤波器组合优化 | 未提供根许可证 |
| RLNN | [source](RLNN%28ICML2025%29/source) | 迭代翻转基线 | 未提供根许可证 |

许可证原文件按上游快照保存；未提供许可证的条目仅记录公开可读状态，不视为已授予通用再分发许可。模型权重和大型数据集不随源码归档。四个补充方法可用现有下载器获取：

~~~powershell
python tools/fetch_references.py all --method SLOG --method S2GNN --method Spectral-benchmark --method MORGAN
~~~
