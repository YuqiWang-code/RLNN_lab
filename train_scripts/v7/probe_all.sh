#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

# Run the census for every reference seed, then apply the pre-registered gates.
refs=(${REFS:-0})
phase="${1:-both}"
shift || true
for ref in "${refs[@]}"; do
  echo "== ref${ref} =="
  REFERENCE_SEED="$ref" bash "$SCRIPTS/probe.sh" "$phase" "$@" 2>&1 | tail -n 3
done
"$PYTHON" "$SCRIPTS/summarize.py" --root "$OUT" --refs "${refs[@]}"
