#!/usr/bin/env bash
set -euo pipefail
base="${RLNN_BASE_PYTHON:-/home/hzeng/envs/rlnn/bin/python}"
target="${MAMBA_ENV:-/home/hzeng/envs/rlnn_mamba}"
"$base" -m venv --system-site-packages "$target"
site=$("$target/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
"$base" -c 'import sysconfig; print(sysconfig.get_path("purelib"))' > "$site/rlnn_base.pth"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13}"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-8}" TORCH_CUDA_ARCH_LIST=12.0 MAMBA_KEEP_CUDA_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE
py="$target/bin/python"
build="$(mktemp -d "$target/mamba-build.XXXXXX")"
cd "$build"
"$py" -m pip download --no-deps --no-build-isolation --no-binary :all: 'mamba-ssm==2.3.2.post1'
tar -xf mamba_ssm-2.3.2.post1.tar.gz
cd mamba_ssm-2.3.2.post1
"$py" - <<'PY'
from pathlib import Path

setup = Path("setup.py")
source = setup.read_text()
start = source.index('        cc_flag.append("-gencode")')
stop = source.index("    # HACK:", start)
setup.write_text(source[:start] + '        cc_flag.extend(["-gencode", "arch=compute_120,code=sm_120"])\n\n' + source[stop:])
PY
if grep -q 'rsqrt(double x);' "$CUDA_HOME/include/crt/math_functions.h" &&
   grep -q '__MATHCALL (rsqrt' /usr/include/x86_64-linux-gnu/bits/mathcalls.h; then
  headers="$build/cuda-include"
  cp -a --reflink=auto "$(readlink -f "$CUDA_HOME/include")" "$headers"
  sed -i -e 's/rsqrt(double x);/rsqrt(double x) noexcept (true);/' \
    -e 's/rsqrtf(float x);/rsqrtf(float x) noexcept (true);/' "$headers/crt/math_functions.h"
  export NVCC_PREPEND_FLAGS="-I$headers"
fi
"$py" -m pip install --no-cache-dir --no-deps --no-build-isolation .
"$py" -m pip install --no-cache-dir 'transformers>=4.45,<5' einops
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
