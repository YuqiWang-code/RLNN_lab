#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
action="${1:?pass correctness, timing or information}"
shift
case "$action" in
  correctness)
    ACCELERATE_USE_CPU=true "$PYTHON" -u "$SCRIPTS/check.py" --device cpu --output "$OUT/preflight/cpu"
    for precision in no fp16; do
      "$PYTHON" -u "$SCRIPTS/check.py" --device cuda --precision "$precision" --output "$OUT/preflight/cuda_$precision"
    done ;;
  timing)
    width="${WIDTH:-32}"
    for precision in no fp16; do
      "$PYTHON" -u models/profile_v7_context.py --precision "$precision" --width "$width" \
        --output "$OUT/preflight/profile_${width}_${precision}${TIMING_TAG:-}" "$@"
    done ;;
  information)
    exec "$PYTHON" -u models/probe_v7_context.py --output "$OUT/preflight/information_probe" "$@" ;;
  *) exit 2 ;;
esac
