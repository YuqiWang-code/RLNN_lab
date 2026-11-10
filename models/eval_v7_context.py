"""Fixed-work and deadline evaluation with identical frozen-B/PAVE pairing."""

import json
from pathlib import Path
from statistics import mean, median, quantiles, stdev

import torch

import eval_v3
from eval_v2 import ROOT, load_policy, write_run
from mis_static_prior import STATIC_FEATURES, STATIC_MODES
from v7_runs import digest, pave, provenance


def main():
    parser = eval_v3.build_parser()
    parser.set_defaults(split="validation", post=["none", "pave"], horizon=1, mixed_precision="no")
    parser.add_argument("--reference_seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--cuda-graph", action="store_true")
    args = parser.parse_args()
    if args.frequency != "none" or set(args.post) - {"none", "pave"} or args.anchored or args.extra_value_checkpoint:
        parser.error("Use original B or a static context with frozen PAVE")
    expected = dict(num_k=20, num_t=100, num_d=5, beta=1.02, horizon=1, candidate_cap=32,
                    exchange_rounds=3, outer_rounds=8, improving_cap=48, repair_rounds=4, post_reserve_s=0.05)
    for key, value in expected.items():
        if getattr(args, key) != value:
            parser.error(f"v7 evaluation fixes {key}={value}")
    if args.time_budget_s is not None:
        if args.time_budget_s not in (0.5, 1.0) or args.post != ["pave"]:
            parser.error("Use one PAVE policy and a 0.5 or 1.0 second budget")
        args.extended_search = True
    if args.split == "test":
        path = ROOT / "outputs/v7_context/analysis/decision.json"
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8"))["status"] != "go_a":
            parser.error("The author test requires a passing validation decision")
    args.backbone_seed = args.reference_seed
    args.head_seed = args.reference_seed
    args.value_checkpoint = pave(args.reference_seed) if "pave" in args.post else None
    args.result_dir.mkdir(parents=True, exist_ok=False)
    model_metadata = {}

    def prepare(options):
        solver = load_policy(options)
        solver.cuda_graph_search = options.cuda_graph
        model = solver.accelerator.unwrap_model(solver.model)
        if solver.v7_context:
            norm = model.adapter
            from hashlib import sha256
            if hasattr(norm, "feature_mean"):
                model_metadata["feature_norm_sha256"] = sha256(norm.feature_mean.cpu().numpy().tobytes()
                                                                + norm.feature_std.cpu().numpy().tobytes()).hexdigest()
            model_metadata["adapter_params"] = sum(p.numel() for p in norm.parameters())
        else:
            model_metadata["adapter_params"] = 0
        return solver

    source = provenance(args.reference_seed)
    if args.context_mode in STATIC_MODES:
        source["static_feature_schema"] = STATIC_FEATURES if args.context_mode == "static_centered" else ()
    rows = eval_v3.evaluate(args, policy_loader=prepare)
    for policy, records in rows.items():
        path = args.result_dir / policy
        data = json.loads((path / "run.json").read_text(encoding="utf-8"))
        sizes = [row["final_size"] for row in records]
        times = [row["solve_time_s"] for row in records]
        data.update(**source, **model_metadata, version="v7_context", arm=args.context_mode,
                    ref=args.reference_seed, model_seed=args.train_seed, order_seed=args.v7_order_seed,
                    checkpoint_sha256=digest(Path(args.save_dir) / "best.pt"), cuda_graph=args.cuda_graph,
                    graph_ids=[row["graph_id"] for row in records], peak_memory_mib=data["peak_memory_mib"],
                    graph_size_sd=stdev(sizes) if len(sizes) > 1 else 0.,
                    p50_solve_s=median(times), p95_solve_s=quantiles(times, n=100, method="inclusive")[94] if len(times) > 1 else times[0],
                    mean_search_steps=mean(row["search_steps"] for row in records),
                    fallback_count=sum(row["incumbent_fallback"] for row in records))
        write_run(path, data)


if __name__ == "__main__":
    main()
