#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
stage="${1:?pass p1, oracle or p2}"
arm="${2:?pass the training arm}"
shift 2
v2_args "$@"
features=8
mode=static_centered
encoder=mlp
direction=()
case "$stage:$arm" in
  p1:centered8) decision=train_p1 ;;
  p1:centered5) decision=train_p1; features=5 ;;
  p1:constants) decision=train_p1; mode=static_constants ;;
  oracle:centered8)
    decision=oracle_direction
    direction=(--oracle_train_targets "${ORACLE_TRAIN_TARGETS:?set path to TRAIN-only KaMIS targets}") ;;
  p2:mlp8) decision=train_p2; encoder=matched-mlp ;;
  p2:msg8) decision=train_p2; encoder=msg ;;
  p2:mamba8) decision=train_p2; encoder=mamba ;;
  p2:mamba8-shuffle) decision=train_p2; encoder=shuffle ;;
  p2:mamba8-degree) decision=train_p2; encoder=degree ;;
  *) echo "Unknown v2 training stage/arm $stage:$arm" >&2; exit 2 ;;
esac
stage_input=p0
[[ "$stage" != p2 ]] || stage_input=p1
suffix=""
[[ "$BACKEND" == fp32-graph ]] || suffix="_$BACKEND"
gate="${GATE_JSON:-$OUT/${stage_input}_summary_ref${REF_SEED}${suffix}.json}"
node_limit="$("$PYTHON" - "$gate" "$decision" "$BACKEND" "${NODE_LIMIT:-}" <<'PY'
import json
import sys
from math import isfinite
from pathlib import Path

path, expected, backend, override = sys.argv[1:]
result = json.loads(Path(path).read_text(encoding="utf-8"))["decisions"].get(backend)
if result is None:
    raise SystemExit(f"{path} has no decision for BACKEND={backend}")
oracle = expected == "oracle_direction"
if oracle and result.get("oracle_dose") is None:
    raise SystemExit(f"{path} has no passing oracle arm for {backend}")
if not oracle and result["status"] != expected:
    raise SystemExit(f"{path} decision is {result['status']}; this stage requires {expected}")
limit = override or result.get("oracle_dose" if oracle else "node_limit")
if limit is None:
    raise SystemExit("Set NODE_LIMIT to the dose supported by the preceding stage")
limit = float(limit)
if not isfinite(limit) or limit <= 0:
    raise SystemExit("The requested dose must be positive and finite")
print(2 * limit if oracle else limit)
PY
)"
export PYTORCH_ALLOC_CONF=backend:cudaMallocAsync
exec "$PYTHON" -u models/train_v7_context.py --static-v2 --context_mode "$mode" --static_encoder "$encoder" \
  --static_features "$features" --static_constant_limit 0.25 --static_node_limit "$node_limit" \
  --reference_seed "$REF_SEED" --result_dir "$OUT/${stage}_${arm}_ref${REF_SEED}" --seed "$SEED" --epochs 3 \
  --graph_batch_size 8 --batch_size 64 --lr 0.0003 --alpha_lr 0.001 --student_mix 0.2 --student_mix_every 1 \
  --method RLNN --loss erdoes --problem mis --beta 1.02 --lambd 0 \
  --train_path "$DATA/RB_small_train" --train_samples 1000 --valid_path "$DATA/RB_small_val" --valid_samples 100 \
  --test_path "$DATA/RB_small_author_test" --num_workers 2 --num_k 20 --num_t 100 --num_tp 50 \
  --num_d 5 --num_l 5 --num_h 128 --v7_width "${WIDTH:-32}" --mixed_precision no "${direction[@]}" "${EXTRA[@]}"
