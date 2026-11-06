# 训练环境

环境路径：`/home/hzeng/envs/nexco`。

[environment.json](environment.json) 保存本次运行的主要包版本，[requirements.txt](requirements.txt) 保存完整包列表。可重建配置为 `models/environment-5090.yml`。

激活环境后，在 `models/diffusion/utils/cython_merge/` 执行 `python setup.py build_ext --inplace`，在 `models/diffusion/utils/local_search/c_classic/` 执行 `make`。RTX 5090 使用 PyTorch 2.9.1+cu130，Lightning 1.9.5 配合 NumPy 1.26.4。
