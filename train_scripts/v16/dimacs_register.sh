#!/usr/bin/env bash
set -euo pipefail
root=${RLNN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$root"
dataset=${1:-models/data/v16_dimacs_register}
out=${2:-outputs/v16/dimacs_register}
core=${CPU_CORE:?set CPU_CORE to the assigned physical core}
mkdir -p "$out"
if [[ ! -e "$out/protocol.json" ]]; then
    cp train_scripts/v16/dimacs_protocol.json "$out/protocol.json"
fi
for arm in degree learned online_mis; do
    taskset -c "$core" bash train_scripts/v16/anytime.sh ER_small "$arm" "$dataset" "$out/$arm" 200 1017 1019 1021
done
