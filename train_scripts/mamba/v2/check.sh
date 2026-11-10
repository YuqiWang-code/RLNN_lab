#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
for script in "$SCRIPTS"/*.sh; do bash -n "$script"; done
"$PYTHON" "$SCRIPTS/check.py"
"$PYTHON" models/eval_static_prior.py v2 --help >/dev/null
"$PYTHON" models/diagnose_static_leverage.py --help >/dev/null
"$PYTHON" models/train_v7_context.py --help >/dev/null
