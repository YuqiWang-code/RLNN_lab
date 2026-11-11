#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

echo "== shell syntax =="
for script in "$SCRIPTS"/*.sh; do bash -n "$script"; done
echo "== module identity and wiring =="
"$PYTHON" "$SCRIPTS/check.py"
echo "== parser surface =="
for entry in models/train_structure.py models/eval_structure.py; do
  "$PYTHON" "$entry" --help >/dev/null
done
"$PYTHON" - <<'PY'
import subprocess
import sys

help_text = subprocess.check_output([sys.executable, "models/main.py", "--help"], text=True)
for flag in ("--structure_mode", "degree", "gate", "gain"):
    assert flag in help_text, flag
print("parser exposes --structure_mode with degree/gate/gain/all")
PY
echo "== frozen backbone present =="
ls -l "$ROOT/outputs/v2/none_seed${REFERENCE_SEED}/checkpoints/best.pt"
echo "all checks passed"
