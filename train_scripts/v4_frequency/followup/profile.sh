#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
OUT="$ROOT/outputs/v4_frequency/followup"
while [[ ! -f "$OUT/queue/branch_fp32.done.txt" ]]; do sleep 10; done
nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/queue/profile_gpu0_before.csv"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u models/profile_v4_frequency.py \
  --split train --seed 0 --graph-count 20 --warmup 30 --repeats 5 \
  --component-iterations 20 --native-forward-repeats 3 --precision fp16 \
  --result-dir "$OUT/p4_profile/train_fp16_seed0" > "$OUT/queue/profile_train_fp16_seed0.log" 2>&1
date -Is > "$OUT/queue/profile.done.txt"
