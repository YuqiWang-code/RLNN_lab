"""Fixed-work and fixed-time v5 evaluation with the frozen v3 PAVE head."""
import json
import math
import os
from pathlib import Path
from statistics import mean

import torch

import eval_v3
from eval_v2 import ROOT, TRAIN, dataset_at, load_policy, search, write_run
from mis_v3_search import SearchGraph
from archived_frequency.mis_v5_frequency import V5_MODES, local_energy, v5_topology
from archived_frequency.train_v5_frequency import digest


def build_parser():
    parser = eval_v3.build_parser()
    parser.set_defaults(split="validation", post=["none", "pave"], horizon=1, num_k=20,
                        num_t=100, mixed_precision="fp16", candidate_cap=32)
    parser.add_argument("--analytic-scale", type=float)
    parser.add_argument("--calibrate-analytic", action="store_true")
    return parser


def configure(args):
    if args.frequency not in ("none",) + V5_MODES or args.anchored or args.extra_value_checkpoint:
        raise ValueError("Use a v5 arm or the original backbone with the frozen H1 protocol")
    if set(args.post) - {"none", "rule23", "pave"}:
        raise ValueError("Postprocessing must be none, rule23 or frozen PAVE")
    expected = dict(horizon=1, outer_rounds=8, repair_rounds=4, improving_cap=48,
                    candidate_cap=32, exchange_rounds=3, num_k=20, num_t=100, num_d=5, beta=1.02)
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError(f"v5 evaluation fixes {name}={value}")
    if args.time_budget_s not in (None, 0.5, 1.0):
        raise ValueError("Use fixed work or a 0.5/1.0 second budget")
    if "pave" in args.post and args.post_reserve_s != 0.05:
        raise ValueError("The frozen PAVE reserve is 0.05 seconds")
    seed = args.train_seed if args.backbone_seed is None else args.backbone_seed
    path = ROOT / f"outputs/v3/stats_h1_seed{seed}/checkpoints/best.pt"
    if args.value_checkpoint is not None and args.value_checkpoint.resolve() != path.resolve():
        raise ValueError("Use the frozen same-seed PAVE checkpoint")
    if args.head_seed not in (None, seed):
        raise ValueError("PAVE and backbone seeds must match")
    args.head_seed = seed
    args.value_checkpoint = path if "pave" in args.post else None


def bind_analytic(solver, scale):
    def topology(graph, adj):
        return v5_topology(graph.edge_index, adj, graph.num_nodes)

    def logits(x, graph, adj, topo):
        base = solver.model(x.unsqueeze(-1), graph.edge_index, topo.normalized).squeeze(-1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            gain = -(local_energy(x, topo.binary @ x, 1.02) / 5).tanh()
        return base + (scale * gain).to(base.dtype)

    solver.policy_topology = topology
    solver.flip_logits = logits


@torch.no_grad()
def calibrate(args):
    if args.frequency != "none" or args.mixed_precision != "no":
        raise ValueError("Analytic calibration uses the FP32 original backbone")
    args.test_path = str(TRAIN)
    args.test_samples = 20
    solver = load_policy(args)
    solver.model.requires_grad_(False)
    dataset = dataset_at(TRAIN, solver.dtype, 20)
    device = solver.accelerator.device
    grid = (0.0, 0.0625, 0.125, 0.25)
    sizes = {scale: [] for scale in grid}
    for index, graph in enumerate(dataset):
        graph_seed = args.seed + index
        cpu_graph = SearchGraph(graph.num_nodes, graph.edge_index, seed=graph_seed)
        graph = graph.to(device)
        for scale in grid:
            bind_analytic(solver, scale)
            selected, _ = search(solver, graph, cpu_graph, graph_seed)
            sizes[scale].append(selected.bit_count())
    means = {scale: mean(values) for scale, values in sizes.items()}
    chosen = max(grid, key=means.__getitem__)
    args.result_dir.mkdir(parents=True, exist_ok=False)
    record = dict(status="completed", split="train", seed=args.train_seed, search_seed=args.seed,
                  samples=len(dataset), graph_ids=[Path(p).stem for p in dataset.files],
                  source_sha=os.environ.get("RLNN_SOURCE_SHA"),
                  backbone_sha256=digest(Path(args.save_dir) / "best.pt"), grid_mean_size=means,
                  sizes=sizes, analytic_scale=chosen)
    write_run(args.result_dir, record)
    print(json.dumps(record, default=str), flush=True)


def evaluate(args):
    configure(args)
    if args.calibrate_analytic:
        return calibrate(args)
    if args.analytic_scale is not None and args.frequency != "none":
        raise ValueError("The analytic correction uses the original backbone")
    args.result_dir.mkdir(parents=True, exist_ok=False)

    def prepare(options):
        solver = load_policy(options)
        if options.analytic_scale:
            bind_analytic(solver, options.analytic_scale)
        return solver

    loader = eval_v3.load_policy
    eval_v3.load_policy = prepare
    try:
        records = eval_v3.evaluate(args)
    finally:
        eval_v3.load_policy = loader
    weight = Path(args.save_dir) / "best.pt"
    for policy, rows in records.items():
        path = args.result_dir / policy
        data = json.loads((path / "run.json").read_text(encoding="utf-8"))
        times = sorted(row["solve_time_s"] for row in rows)
        data.update(version="v5_frequency", frequency=args.frequency, source_sha=os.environ.get("RLNN_SOURCE_SHA"),
                    checkpoint_sha256=digest(weight), graph_ids=[row["graph_id"] for row in rows],
                    pave_sha256=digest(args.value_checkpoint) if policy == "pave" else None,
                    pave_protocol="v3_stats_h1_frozen" if policy == "pave" else None,
                    trainable_params=0, analytic_scale=args.analytic_scale, p95_time_s=times[math.ceil(.95 * len(times)) - 1],
                    mean_search_steps=mean(row["search_steps"] for row in rows),
                    max_deadline_overrun_s=max(0.0, times[-1] - args.time_budget_s) if args.time_budget_s else 0.0)
        write_run(path, data)
    return records


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
