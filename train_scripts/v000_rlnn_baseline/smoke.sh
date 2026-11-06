#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
cd "$ROOT/models"
exec "${PYTHON:-python}" "$ROOT/tools/run_rlnn.py" smoke --run "${RUN_NAME:-smoke_$(date +%Y%m%d_%H%M%S)}" \
  --method RLNN --loss erdoes --num_l 5 --num_h 128 "$@"
