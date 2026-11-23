#!/usr/bin/env bash
# Rank-composition ablation at the 200 ms deadline.
#
# The shipped priority is p_v = z_G(-d)_v + 0.25 * z_G(g_theta(G))_v. This
# launcher re-runs the same feasible search with other mixes and with the
# learned score alone, so the marginal contribution of the learned term can be
# read against the degree-only arm. Everything except the ranking is fixed by
# train_scripts/v15/priority_200ms.sh.
set -euo pipefail
root=${RLNN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$root"
out=${1:?usage: priority_ranks_ablation.sh OUTPUT_ROOT [ARM ...]}
shift
arms=("$@")
if (( ${#arms[@]} == 0 )); then arms=(learned:0.1 learned:1.0 neural); fi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
for arm in "${arms[@]}"; do
    tag=${arm/:/_}
    bash train_scripts/v15/priority_200ms.sh RB_small "$arm" models/data/MIS_tensors/RB_small_author_test "$out/RB_small/$tag"
    bash train_scripts/v15/priority_200ms.sh RB_large "$arm" models/data/MIS_tensors/RB_large_author_test "$out/RB_large/$tag"
    bash train_scripts/v15/priority_200ms.sh ER_small "$arm" models/data/MIS_tensors/ER_small_author_test "$out/ER_small/$tag"
    bash train_scripts/v15/priority_200ms.sh ER_small "$arm" models/data/MIS_tensors/ER_large_author_test "$out/ER_large/$tag"
done
