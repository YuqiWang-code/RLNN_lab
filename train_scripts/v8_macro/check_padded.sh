#!/usr/bin/env bash
# Gate the amortised padded engine: bit-identical trajectory, replay and decode.
#
# Passes when the padded buffers reproduce the unpadded frozen loop, the captured
# replay reproduces the padded eager loop, and the packed decode matches
# solution_masks.  Cheap: --graphs 4 --steps 60 is a few seconds.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
export CUDA_VISIBLE_DEVICES="${V8_GPU:-1}"

mkdir -p "$OUT/queue"
"$PYTHON" -u models/check_padded_capture.py \
  --reference_seed "$REFERENCE_SEED" --save_dir "$BACKBONE" --split validation \
  --test_samples "${GRAPHS:-4}" --steps "${STEPS:-60}" --chunk "${CHUNK:-10}" \
  --seeds "${SEEDS_LIST:-1017}" --decode_chunk "${DECODE_CHUNK:-4}" \
  --result_dir "$OUT/padded" 2>&1 | tee -a "$OUT/queue/check_padded.log"
