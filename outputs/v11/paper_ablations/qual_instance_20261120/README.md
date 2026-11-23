# First RB test instance

`case.json` records the undirected edge list and the best A.3-decoded set from the four lowest-energy particle archives after each ten-step block, beginning at initialization. Vertex identifiers use zero-based tensor indices. The example uses the first test graph and first search seed, 1017. The 350-step horizon matches the observed block count for this pair in the midpoint-decoding timing study. The diagnostic inspects 36 checkpoints in a separate fixed-work run. Its outputs support the membership and conflict panels of the figure.

Run the diagnostic from the repository root on the experiment server.

```bash
shape=outputs/v11/paper_ablations/cuda_graph_compile_20261119/fixed130/full/batch_1017/seed_1017/padded/padded_ref0.json
out=outputs/v11/paper_ablations/qual_instance_20261120/case.json
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=models python tools/capture_qualitative_case.py \
  --case_file "$out" --shape_summary "$shape" --result_dir "$(dirname "$out")" \
  --split test --test_path models/data/MIS_tensors/RB_small_author_test --test_samples 1000 \
  --reference_seed 0 --save_dir outputs/v2/none_seed0/checkpoints --num_k 200 --chunk 10 \
  --mixed_precision fp16 --skip_incumbent --compile_model --dense --no_pave \
  --decoders captured --decode_every 1000 --decode_top 4 --seeds 1017
```

`python tools/plot_deadline_paper.py` writes the vector figures to `overleaf/figures/` using Matplotlib, NetworkX, and NumPy. The availability curve uses the original timing logs in `decode_schedule_20261119`, and the membership and conflict panels use `case.json`.
