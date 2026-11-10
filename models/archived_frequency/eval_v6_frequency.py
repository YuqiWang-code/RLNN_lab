"""Evaluate SC-RLNN and the reference policies with frozen H1 PAVE."""

import json
import os
from statistics import mean, stdev

import eval_v3
from eval_v2 import ROOT, write_run
from archived_frequency.mis_v5_frequency import V5_MODES
from archived_frequency.mis_v6_frequency import V6_MODES


def main():
    parser = eval_v3.build_parser()
    parser.set_defaults(split="validation", post=["none", "pave"], horizon=1)
    parser.add_argument("--reference-seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--cuda-graph", action="store_true")
    args = parser.parse_args()
    if args.frequency not in ("none",) + V5_MODES + V6_MODES:
        parser.error("Use Backbone B, a v5 control or a v6 arm")
    if set(args.post) - {"none", "pave"} or args.anchored or args.extra_value_checkpoint:
        parser.error("Use no postprocessing or the frozen H1 PAVE protocol")
    expected = dict(num_k=20, num_t=100, num_d=5, beta=1.02, horizon=1, candidate_cap=32,
                    exchange_rounds=3, outer_rounds=8, improving_cap=48, repair_rounds=4, post_reserve_s=0.05)
    for name, value in expected.items():
        if getattr(args, name) != value:
            parser.error(f"v6 evaluation fixes {name}={value}")
    if args.time_budget_s is not None:
        if args.time_budget_s not in (0.5, 1.0) or args.post != ["pave"]:
            parser.error("Fixed-time evaluation uses frozen PAVE with 0.5 or 1.0 seconds")
        args.extended_search = True
    args.head_seed = args.reference_seed
    if "pave" in args.post:
        args.value_checkpoint = ROOT / f"outputs/v3/stats_h1_seed{args.reference_seed}/checkpoints/best.pt"
    args.result_dir.mkdir(parents=True, exist_ok=False)
    loader = eval_v3.load_policy

    def load(options):
        solver = loader(options)
        solver.cuda_graph_search = options.cuda_graph
        return solver

    eval_v3.load_policy = load
    try:
        records = eval_v3.evaluate(args)
    finally:
        eval_v3.load_policy = loader
    for policy, rows in records.items():
        folder = args.result_dir / policy
        data = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        sizes = [row["final_size"] for row in rows]
        data.update(version="v6", frequency=args.frequency, reference_seed=args.reference_seed,
                    source_sha=os.environ.get("RLNN_SOURCE_SHA"),
                    sd_graph_size=stdev(sizes) if len(sizes) > 1 else 0,
                    mean_search_steps=mean(row["search_steps"] for row in rows),
                    graph_ids=[row["graph_id"] for row in rows])
        write_run(folder, data)


if __name__ == "__main__":
    main()
