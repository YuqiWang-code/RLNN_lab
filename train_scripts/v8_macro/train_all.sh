#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
for arm in A2 A3 A5; do
  echo "== $arm =="
  EPOCHS="${EPOCHS:-3}" bash "$(dirname -- "${BASH_SOURCE[0]}")/train.sh" "$arm"
done
