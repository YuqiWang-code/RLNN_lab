"""Evaluate frequency adapters with the frozen v3 statistics head."""
import json
from pathlib import Path

import eval_v3
from eval_v2 import ROOT, load_policy, write_run
from mis_v4_frequency import V4_MODES


def build_parser():
    parser = eval_v3.build_parser()
    parser.set_defaults(post=["none", "pave"], horizon=1, num_k=20, num_t=100,
                        mixed_precision="fp16", candidate_cap=32)
    parser.add_argument("--disable-low", action="store_true")
    parser.add_argument("--disable-high", action="store_true")
    parser.add_argument("--disable-gate", action="store_true",
                        help="Use constant 0.5 gates for both adapter branches")
    return parser


def configure(args):
    """Bind v4 to the same-seed frozen v3 PAVE protocol."""
    if args.frequency not in ("none",) + V4_MODES:
        raise ValueError("v4 evaluation requires a v4 frequency arm or none")
    if set(args.post) - {"none", "rule23", "pave"} or args.extra_value_checkpoint:
        raise ValueError("v4 evaluates none, rule23 and the frozen statistics PAVE head")
    expected = dict(horizon=1, outer_rounds=8, repair_rounds=4, improving_cap=48,
                    candidate_cap=32, exchange_rounds=3, num_k=20, num_t=100)
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError(f"The v4 protocol fixes {name}={value}")
    if args.anchored:
        raise ValueError("v4 uses the original H1 action space")
    if args.time_budget_s is not None and args.time_budget_s != 0.5:
        raise ValueError("The v4 fixed-time budget is 0.5 seconds")
    if args.post_reserve_s != 0.05:
        raise ValueError("The frozen PAVE reserve is 0.05 seconds")
    seed = args.train_seed if args.backbone_seed is None else args.backbone_seed
    checkpoint = ROOT / f"outputs/v3/stats_h1_seed{seed}/checkpoints/best.pt"
    if args.value_checkpoint is not None and args.value_checkpoint.resolve() != checkpoint.resolve():
        raise ValueError(f"Use the frozen same-seed PAVE checkpoint: {checkpoint}")
    if args.head_seed is not None and args.head_seed != seed:
        raise ValueError("The PAVE head seed must match the backbone seed")
    args.value_checkpoint = checkpoint if "pave" in args.post else None
    args.head_seed = seed
    if args.frequency == "none" and any((args.disable_low, args.disable_high, args.disable_gate)):
        raise ValueError("Adapter interventions require a frequency adapter")
    return args


def prepare(args):
    solver = load_policy(args)
    model = solver.accelerator.unwrap_model(solver.model)
    if args.frequency != "none":
        model.adapter.low_override = False if args.disable_low else None
        model.adapter.high_override = False if args.disable_high else None
        model.adapter.gate_override = 0.5 if args.disable_gate else None
    return solver


def evaluate(args):
    configure(args)
    original_loader = eval_v3.load_policy
    eval_v3.load_policy = prepare
    try:
        records = eval_v3.evaluate(args)
    finally:
        eval_v3.load_policy = original_loader
    for policy, rows in records.items():
        directory = args.result_dir / policy
        metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        metadata.update(
            version="v4_frequency", frequency=args.frequency,
            backbone_checkpoint=str(Path(args.save_dir) / "best.pt"),
            pave_protocol="v3_stats_h1_frozen" if policy == "pave" else None,
            interventions=dict(disable_low=args.disable_low, disable_high=args.disable_high,
                               constant_half_gates=args.disable_gate),
            trainable_params=0,
            mean_search_steps=sum(row["search_steps"] for row in rows) / len(rows),
            max_solve_time_s=max(row["solve_time_s"] for row in rows),
            max_deadline_overrun_s=(max(0.0, max(row["solve_time_s"] for row in rows) - args.time_budget_s)
                                    if args.time_budget_s is not None else 0.0),
        )
        write_run(directory, metadata)
    return records


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
