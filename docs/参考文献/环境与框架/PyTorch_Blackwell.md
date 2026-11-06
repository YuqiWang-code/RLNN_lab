# PyTorch 与 Blackwell

[PyTorch 2.7 官方发布说明](https://pytorch.org/blog/pytorch-2-7/)明确列出 NVIDIA Blackwell 支持与 CUDA 12.8 预编译包。

项目服务器使用 RTX 5090。原始 `models/environment.yml` 固定了 PyTorch 1.11.0+cu113，环境适配需要同时处理 Python、Lightning 和 PyG 扩展版本。完整训练环境以小规模训练与测试通过后的固定配置为准。
