"""One capture for the whole dataset: the amortised prelude arm.

The per-graph prelude of ``eval_v8_capture_budget`` is a full CUDA graph capture
of the search block, and it costs about 93 ms -- half of a 0.2 s clock.  Here the
capture happens once, over padded buffers, and each graph only rewrites the edge
buffers (a few milliseconds).  Everything else (frozen backbone, A.3 decode,
PAVE, the clock accounting) is identical, so the two tables are comparable.
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from diagnose_v7_census import build_parser, solution_masks
from eval_v2 import dataset_at, load_policy, sync
from eval_v7_budget import POST_RESERVE, run_pave
from eval_v8_capture_budget import build_capture_parser, paired_bootstrap
from mis_v1 import adjacency
from mis_v3_search import SearchGraph
from mis_v3_value import PAVEV3
from mis_v7_recombine import is_independent
from mis_v8_pad import PaddedSearch
from v7_runs import pave


def shape_of(dataset, limit=None):
    """Largest node count and largest out-degree in the split."""
    nodes = degree = 0
    for index in range(len(dataset) if limit is None else min(limit, len(dataset))):
        graph = dataset[index]
        edges = graph.edge_index
        nodes = max(nodes, int(graph.num_nodes))
        degree = max(degree, int(torch.bincount(edges[0], minlength=nodes).max()))
    return nodes, degree + 1


def build_parser():
    parser = build_capture_parser()
    parser.add_argument("--reserve_frac", type=float, default=0.2,
                        help="PAVE reserve as a fraction of the budget (capped by --post_reserve)")
    parser.add_argument("--padded_shapes", type=int, default=None,
                        help="cap the shape scan (default: the whole split)")
    parser.set_defaults(arms=["padded"], decoders=["captured", "cached"])
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    output = args.result_dir / "padded"
    output.mkdir(exist_ok=True)

    solver = load_policy(args)
    solver.model.requires_grad_(False)
    checkpoint = torch.load(pave(args.reference_seed), map_location="cpu", weights_only=False)
    ranker = PAVEV3(checkpoint["mode"], checkpoint.get("stats_dim", 24))
    ranker.load_state_dict(checkpoint["model_state"])
    ranker = ranker.cpu().eval()
    device = solver.accelerator.device
    dataset = dataset_at(args.test_path, solver.dtype, args.test_samples,
                         official=args.split == "test")
    indices = list(range(len(dataset)))[args.shard::args.shards]
    n_max, capacity = shape_of(dataset, args.padded_shapes)
    print(f"padded shape: nodes {n_max}, capacity {capacity}", flush=True)

    entered = perf_counter()
    engine = PaddedSearch(solver, n_max=n_max, capacity=capacity, chunk=args.chunk,
                          device=device, decode_chunk=args.decode_chunk)
    capture_seconds = perf_counter() - entered
    # One warm replay so the first measured cell meets a warm allocator.
    engine.load(dataset[0].to(device), dataset[0].edge_index, args.seeds[0])
    engine.run(perf_counter() + 0.05, lambda states: None)
    sync(device)

    extended = 20000
    solver.num_t = extended
    cells = [(arm, kind) for arm in args.arms for kind in args.decoders]
    rows = []
    started = perf_counter()
    for position, index in enumerate(indices):
        graph = dataset[index]
        graph_id = f"idx_{index}" if args.split == "test" else Path(dataset.files[index]).stem
        edges = graph.edge_index
        n = int(graph.num_nodes)
        graph = graph.to(device)
        adj = adjacency(graph)
        for seed in args.seeds:
            for budget in args.budgets:
                topology = SearchGraph(n, edges, seed=seed)
                neighbours = topology.neighbors
                order = cells if (index + seed) % 2 == 0 else cells[::-1]
                for repeat in range(args.repeats):
                    arms = order if repeat == 0 else order[::-1]
                    for arm, decoder_kind in arms:
                        sync(device)
                        tick = perf_counter()
                        deadline = tick + budget
                        reserve = min(args.post_reserve, args.reserve_frac * budget)
                        cutoff = deadline - reserve
                        incumbent = 0
                        masks_box = []

                        def decode(best, kind=decoder_kind):
                            if kind == "eager":
                                candidate = max(solution_masks(solver.problem, best[:n], adj, graph.b),
                                                key=int.bit_count)
                            else:
                                candidate = engine.decode_best(best)
                            if candidate.bit_count() > incumbent:
                                masks_box.append(candidate)

                        loaded = engine.load(graph, edges, seed)
                        decode(engine.best_states)
                        prelude = perf_counter() - tick
                        steps, decode_seconds, blocks = engine.run(
                            cutoff, decode, skip_unchanged=decoder_kind == "cached")
                        sync(device)
                        search_seconds = perf_counter() - tick
                        selected = max(masks_box, key=int.bit_count)
                        final = run_pave(topology, selected, ranker, args, deadline)
                        sync(device)
                        seconds = perf_counter() - tick
                        if not is_independent(final, neighbours):
                            raise ValueError(f"{graph_id}: illegal set in {arm}/{decoder_kind}")
                        cell_name = f"{arm}/{decoder_kind}"
                        rows.append(dict(graph_id=graph_id, search_seed=seed, budget=budget,
                                         arm=arm, decoder=decoder_kind, cell=cell_name,
                                         repeat=repeat, first=int(arms[0] == (arm, decoder_kind)),
                                         final_size=final.bit_count(),
                                         search_size=selected.bit_count(),
                                         pave_gain=final.bit_count() - selected.bit_count(),
                                         steps=steps, blocks=blocks,
                                         solve_time_s=round(seconds, 4),
                                         search_time_s=round(search_seconds, 4),
                                         decode_time_s=round(decode_seconds, 4),
                                         load_s=round(loaded, 4),
                                         reserve_s=round(reserve, 4),
                                         prelude_s=round(prelude, 4),
                                         ms_per_step=round(1e3 * search_seconds / max(steps, 1), 3),
                                         deadline_overrun=int(seconds > budget)))
        if (position + 1) % 20 == 0 or position + 1 == len(indices):
            parts = []
            for budget in args.budgets:
                for arm, kind in cells:
                    chosen = [r for r in rows if r["budget"] == budget
                              and r["cell"] == f"{arm}/{kind}"]
                    if chosen:
                        parts.append(f"{budget}s/{arm}/{kind}:"
                                     f"{mean(r['final_size'] for r in chosen):.3f}"
                                     f"@{mean(r['steps'] for r in chosen):.0f}st")
            print(f"{position + 1}/{len(indices)} " + " ".join(parts), flush=True)

    names = [f"{arm}/{kind}" for arm, kind in cells]
    summary = dict(ref=args.reference_seed, seeds=args.seeds, budgets=args.budgets,
                   graphs=len(indices), repeats=args.repeats, horizon=args.horizon,
                   post_reserve=args.post_reserve, chunk=args.chunk, arms=list(args.arms),
                   decoders=list(args.decoders), decode_chunk=args.decode_chunk, cells=names,
                   padded=dict(nodes=n_max, capacity=capacity,
                               capture_s=round(capture_seconds, 4),
                               load_s=round(mean(r["load_s"] for r in rows), 4)),
                   wall_time_s=round(perf_counter() - started, 2))
    if args.shards > 1:
        summary["shard"] = dict(index=args.shard, count=args.shards)

    for budget in args.budgets:
        for name in names:
            chosen = [r for r in rows if r["budget"] == budget and r["cell"] == name]
            if not chosen:
                continue
            print(f"{budget:>4}s {name:<18} size {mean(r['final_size'] for r in chosen):.4f} "
                  f"steps {mean(r['steps'] for r in chosen):.1f} "
                  f"search {mean(r['search_size'] for r in chosen):.4f} "
                  f"ms/step {mean(r['ms_per_step'] for r in chosen):.3f} "
                  f"load {mean(r['load_s'] for r in chosen) * 1e3:.2f} ms "
                  f"decode {mean(r['decode_time_s'] for r in chosen) * 1e3:.2f} ms", flush=True)

    # Paired difference against the eager arm of the same budget, when present.
    for budget in args.budgets:
        base = f"padded/{args.decoders[0]}"
        for name in names[1:]:
            diffs = []
            for graph_id in sorted({r["graph_id"] for r in rows}):
                left = [r for r in rows if r["graph_id"] == graph_id and r["budget"] == budget
                        and r["cell"] == name]
                right = [r for r in rows if r["graph_id"] == graph_id and r["budget"] == budget
                         and r["cell"] == base]
                if left and right:
                    diffs.append(mean(r["final_size"] for r in left)
                                 - mean(r["final_size"] for r in right))
            if diffs:
                gap, low, high = paired_bootstrap(diffs)
                print(f"{budget}s {name} - {base}: {gap:+.4f} [{low:+.4f},{high:+.4f}]", flush=True)

    tag = "" if args.shards == 1 else f"_shard{args.shard}"
    path = output / f"padded_ref{args.reference_seed}{tag}"
    columns = list(rows[0])
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[key] for key in columns])
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {path.with_suffix('.csv')}", flush=True)


if __name__ == "__main__":
    main()
