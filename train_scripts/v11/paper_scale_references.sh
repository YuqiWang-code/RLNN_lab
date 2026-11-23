#!/usr/bin/env bash
set -euo pipefail

regime=${1:?RB_large, ER_small or ER_large}
arm=${2:?reference method}
output_root=${3:-outputs/v11/scale_deadline_20261120}
dataset="models/data/MIS_tensors/${regime}_author_test"
run="$output_root/$regime/$arm"
count=${GRAPHS:--1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/models"
learned="$PWD/tmp/learned_baselines_20261119"
supervised="$PWD/tmp/supervised_baselines_20261119"
lwd="$PWD/tmp/lwd_intel_20261120"
jax_py="$HOME/envs/rlnn_mamba/bin/python"
py="$HOME/envs/rlnn/bin/python"
entry=${REFERENCE_ENTRY:-models/eval_reference_deadline.py}
opts=()
method=$arm
case "$arm" in
    author_rlnn|fld_eg|rlsa) ;;
    author_rlnn_matched)
        method=author_rlnn
        checkpoint=rlnn_er_mis
        if [[ "$regime" == RB_large ]]; then checkpoint=rlnn_rb_large_mis; fi
        opts=(--checkpoint "models/data/author_rld4co/checkpoints/erdoes/$checkpoint")
        ;;
    isco|isco_full_window)
        method=isco
        if [[ "$arm" == isco ]]; then opts=(--isco-max-steps 300); fi
        export PYTHONPATH="$PYTHONPATH:$PWD/tmp/reference_deadline_20261119/discs_source"
        py=$jax_py
        ;;
    diffuco)
        export PYTHONPATH="$PYTHONPATH:$learned/DiffUCO:$learned/deps_diffuco"
        if [[ "$regime" == ER_large ]]; then
            export XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}
        fi
        opts=(--checkpoint "$learned/DiffUCO/Checkpoints/m3h9mz5g/best_m3h9mz5g.pickle")
        py=$jax_py
        ;;
    ltft)
        stage=$learned
        export PYTHONPATH="$PYTHONPATH:$stage/LTFT/gflownet:$stage/deps_ltft" DGLBACKEND=pytorch
        export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
        source "$stage/ltft_cuda_env.sh"
        opts=(--checkpoint "$stage/ltft_training/alg.pt" --ltft-config "$stage/ltft_training/train_config.yaml")
        py=$jax_py
        ;;
    difusco)
        export PYTHONPATH="$PYTHONPATH:$supervised/DIFUSCO:$supervised/deps"
        opts=(--checkpoint "$supervised/mis_sat_categorical.ckpt")
        py=$jax_py
        ;;
    dgl_treesearch)
        stage=$supervised
        source "$stage/supervised_dgl_environment.sh"
        mapfile -t weights < <(find "$stage/dgl_dense_training" -name '*_final_model32.torch')
        [[ ${#weights[@]} == 1 ]]
        opts=(--checkpoint "${weights[0]}")
        py=$jax_py
        ;;
    intel) opts=(--checkpoint "$lwd/intel.npz") ;;
    ppo_lwd)
        stage=$supervised
        export PYTHONPATH="$PYTHONPATH:$lwd/lwd"
        source "$stage/supervised_dgl_environment.sh"
        opts=(--checkpoint "$PWD/outputs/v11/intel_ppo_deadline_20261120/ppo_training/actor.pt")
        py=$jax_py
        ;;
    *) echo "Unknown method: $arm" >&2; exit 2 ;;
esac
mkdir -p "$(dirname "$run")"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv > "$run.log"
if [[ -n ${XLA_PYTHON_CLIENT_ALLOCATOR:-} ]]; then
    echo "XLA_PYTHON_CLIENT_ALLOCATOR=$XLA_PYTHON_CLIENT_ALLOCATOR" >> "$run.log"
fi
"$py" -u "$entry" "$method" --dataset "$dataset" --graphs "$count" --output "$run" "${opts[@]}" 2>&1 | tee -a "$run.log"
