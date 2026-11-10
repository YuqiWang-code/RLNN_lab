#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
OUT="$ROOT/outputs/v4_frequency/followup"
run() {
  local name="$1" gpu="$2"; shift 2
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$@" > "$OUT/queue/$name.log" 2>&1
  local code=$?
  echo "$code" > "$OUT/queue/$name.exit_code.txt"
  return "$code"
}
for seed in 0 1 2; do
  run "reliability_seed$seed" "$((seed % 2))" models/archived_frequency/probe_v4_utility_followup.py --seed "$seed" --split train --graphs 10 --nodes 16 --replicates 16 --sampling uniform --phases 25 75 --horizon 20 --result-dir "$OUT/p3_utility/reliability_seed$seed" &
done
wait
for seed in 0 1 2; do
  run "groups_uniform_seed$seed" 0 models/archived_frequency/probe_v4_utility_followup.py --seed "$seed" --split train --graphs 20 --nodes 16 --replicates 8 --sampling uniform --phases 25 50 100 --horizon 20 --result-dir "$OUT/p3_utility/groups_uniform_seed$seed" &
  run "groups_stratified_seed$seed" 1 models/archived_frequency/probe_v4_utility_followup.py --seed "$seed" --split train --graphs 20 --nodes 24 --replicates 8 --sampling stratified --phases 25 50 100 --horizon 20 --result-dir "$OUT/p3_utility/groups_stratified_seed$seed" &
done
wait
for source in teacher own; do
  if [[ "$source" == teacher ]]; then gpu=0; else gpu=1; fi
  for model in v4a v4a_message v4a_mlp; do
    run "p1_train_${model}_${source}_seed0" "$gpu" models/archived_frequency/diagnose_v4_spectrum.py --seed 0 --split train --graphs 20 --models "$model" --trajectory "$source" --result-dir "$OUT/p1_spectrum/train_${model}_${source}_seed0" &
  done
done
wait
run p2_train_alpha_seed0 0 models/archived_frequency/diagnose_v4_branch_replay.py --seed 0 --split train --graphs 10 --precision fp32 --alpha-scales 0 .25 .5 1 1.5 2 --interventions full no_low no_high gate_half no_adapter high_bias1 high_bias2 high_identity high_no_norm --result-dir "$OUT/p2_branch/train_alpha_seed0"
date -Is > "$OUT/queue/explore.done.txt"
