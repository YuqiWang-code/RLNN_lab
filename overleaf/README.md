# SPRINT: Feasible-Set-First Neural Search for Maximum Independent Set Under a 200-Millisecond Deadline

论文把两条求解路线组成的求解器族称为 **SPRINT**：构造路线（先发布可行集、再用学习排序交换与分支）记为 **SPRINT-C**，投机路线（GPU 粒子搜索与中点/末尾 A.3 解码）记为 **SPRINT-S**。正文、表 1、图 2 与附录统一使用这两个名字；图 2 面板内保留英文描述性标题 “Feasible search”/“Particle search”，由图注映射到 SPRINT-C/SPRINT-S。

将工程上传到 Overleaf，选择 `main.tex` 为主文档，使用 pdfLaTeX 和 BibTeX 编译。论文采用 ICML 2027 样式，并以 `preprint` 选项编译，因此首页显示作者（曾涵 / 张欣，江南大学）与通讯作者信息；若需匿名投稿版，去掉 `\usepackage[preprint]{icml2027}` 中的选项并恢复匿名作者块即可。 [官方模板](https://media.icml.cc/Conferences/ICML2027/styles/icml2027.zip) · [作者说明](https://icml.cc/Conferences/2027/AuthorInstructions)

正文在 `sections/`，参考文献在 `references.bib`。附录说明训练和迁移条件、计时边界、按图重采样的方法与整批成本。 [方法](sections/method.tex) · [实验](sections/experiments.tex) · [四档定量表](sections/scale_table.tex) · [四档定性对照](sections/qualitative_cross_scale.tex) · [限时 KaMIS 与分布权重对照](sections/scale_supplement.tex) · [附录](sections/appendix.tex)

主表与预算曲线使用同一组方法色。蓝色表示 RLNN 优先级，绿色表示度数优先级，橙色表示常驻 OnlineMIS。表格的表头、线宽和底色在 [table_style.tex](table_style.tex) 中定义，绘图脚本共用仓库中的 `tools/paper_figure_style.py`。

Method 的双路线模型图使用仓库 `docs/模型图.pptx` 第 5 页的可编辑母版，[论文矢量图](figures/search_paths.pdf) 从该页导出。引言的最优解到达曲线和配对增益图由 `tools/plot_intro_evidence.py` 从逐图记录和离线最优值重建。常驻求解器的预算曲线、ER 同核复测和明细表由 `tools/summarize_resident_deadlines.py` 生成。执行消融、粒子成员变化与附录逐图解码时钟由 `tools/plot_deadline_paper.py` 生成。基线配对图与冷链算例分别使用 `tools/plot_paper_baseline_pairs.py` 和 `tools/plot_dispatch_example.py`。
