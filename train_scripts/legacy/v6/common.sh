#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/../v4_frequency/common.sh"
OUT="$ROOT/outputs/v6"
REF_SEED="${REF_SEED:-0}"
seeds=(42 123 999)
SEED="${seeds[$REF_SEED]}"
