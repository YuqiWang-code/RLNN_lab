#!/usr/bin/env bash
set -euo pipefail
base="${RLNN_BASE_PYTHON:-/home/hzeng/envs/rlnn/bin/python}"
target="${MAMBA_ENV:-/home/hzeng/envs/rlnn_mamba}"
"$base" -m venv --system-site-packages "$target"
site=$("$target/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
"$base" -c 'import sysconfig; print(sysconfig.get_path("purelib"))' > "$site/rlnn_base.pth"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13}"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-3}" TORCH_CUDA_ARCH_LIST=12.0 MAMBA_KEEP_CUDA_BUILD=TRUE
"$target/bin/python" -m pip install --no-cache-dir --no-build-isolation 'mamba-ssm==2.3.2.post1'
"$target/bin/python" - <<'PY'
import mamba_ssm
import selective_scan_cuda
import torch

x = torch.randn(1, 256, 32, device="cuda", requires_grad=True)
model = mamba_ssm.Mamba(32, d_state=16, d_conv=4, expand=2).cuda()
model(x).square().mean().backward()
assert torch.isfinite(x.grad).all()
print(torch.__version__, torch.version.cuda, mamba_ssm.__version__)
PY
