# 参考实现

[方法清单](catalog.tsv)给出作者仓库、论文和 v1 用途。每种方法的上游文件放在自己的 source 子目录，upstream.json 记录对应版本。RLSA 与 RLNN 使用同一上游项目。

~~~powershell
python tools/fetch_references.py all
~~~

下载只取公开的源码和论文。源码快照保存在本地与实验服务器，Git 跟踪清单、版本和本项目的适配入口。下载器过滤上游打包的权重、数据缓存和编译产物。已公开的部分实现按原仓库归档，IDG 的公开版本是 README 占位。

正式对照的训练和评估入口见 [v1 实验](../train_scripts/v1/README.md)。论文 PDF 按研究方向收在[参考文献](../docs/参考文献/文献索引.md)。
