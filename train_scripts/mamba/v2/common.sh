#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/../v0/common.sh"
SCRIPTS="$ROOT/train_scripts/mamba/v2"
OUT="${OUT_V2:-$ROOT/outputs/mamba_v2}"
BACKEND="${BACKEND:-fp32-graph}"

v2_args() {
  EXTRA=()
  while (($#)); do
    if [[ "$1" == --gpu ]]; then
      export CUDA_VISIBLE_DEVICES="${2:?pass one GPU index}"
      shift 2
    else
      EXTRA+=("$1")
      shift
    fi
  done
}
