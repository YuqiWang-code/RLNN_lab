#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
export PYTORCH_ALLOC_CONF=backend:native
exec "$PYTHON" -u "$ROOT/models/eval_v2.py" "$@"
