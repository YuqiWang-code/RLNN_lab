#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT="$ROOT/outputs/v4_frequency/followup"
mkdir -p "$OUT/queue"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" train_scripts/legacy/v4_frequency/check_spectrum.py --device cpu --output "$OUT/p0_checks/check_spectrum_cpu.json" > "$OUT/queue/check_spectrum_cpu.log" 2>&1
CUDA_VISIBLE_DEVICES=0 "$PYTHON" train_scripts/legacy/v4_frequency/check_spectrum.py --device cuda --output "$OUT/p0_checks/check_spectrum_cuda.json" > "$OUT/queue/check_spectrum_cuda.log" 2>&1
run() {
  local name="$1" gpu="$2"; shift 2
  (CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$@" > "$OUT/queue/$name.log" 2>&1; echo 0 > "$OUT/queue/$name.exit_code.txt")
}
run smoke_p3 1 models/archived_frequency/probe_v4_utility_followup.py --split train --graphs 3 --seed 0 --nodes 8 --replicates 2 --phases 25 --horizon 20 --result-dir "$OUT/p3_utility/smoke_seed0" &
p3=$!
run smoke_p2 1 models/archived_frequency/diagnose_v4_branch_replay.py --split train --graphs 3 --seed 0 --precision fp32 --result-dir "$OUT/p2_branch/smoke_seed0" &
p2=$!
run smoke_p1 0 models/archived_frequency/diagnose_v4_spectrum.py --split train --graphs 3 --seed 0 --models v4a --trajectory teacher --phases 0 25 100 --result-dir "$OUT/p1_spectrum/smoke_f4_teacher_seed0"
run smoke_p4 0 models/archived_frequency/profile_v4_frequency.py --split train --seed 0 --graph-count 2 --warmup 30 --repeats 1 --component-iterations 5 --precision fp16 --result-dir "$OUT/p4_profile/smoke_seed0"
wait "$p3"
wait "$p2"
date -Is > "$OUT/queue/smoke.done.txt"
