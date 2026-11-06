# PyTorch 与 Blackwell

[PyTorch 2.7 官方发布说明](https://pytorch.org/blog/pytorch-2-7/)列出 NVIDIA Blackwell 支持和 CUDA 12.8 预编译包。[PyG 安装说明](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)提供按 PyTorch / CUDA 组合选择扩展 wheel 的规则。

RTX 5090 的待验证组合及安装命令见[服务器说明](../../服务器.md)。上游原始环境固定 PyTorch 2.4.1 / CUDA 12.4，不能直接作为本机型已验证环境。须以真实 RLNN 前向、反向、CSR 与 scatter 运算通过为准。
