#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
export PYTHONPATH="$ROOT/models${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -u models/archived_frequency/profile_v5_frequency.py --seed "${SEED:-0}" --output "$OUT/${RUN_NAME:-profile_seed0}" "$@"
