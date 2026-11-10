#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec "$PYTHON" -u models/diagnose_v3.py --mode collect --workers "${WORKERS:-24}" "$@"
