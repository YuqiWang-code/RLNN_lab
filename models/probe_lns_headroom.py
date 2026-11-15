"""Is there headroom for *large* moves?  Exact repair of a ball, measured.

The exchange probe (``probe_exchange_headroom.py``) showed that the deployed
solution is closed under every move of at most four points.  This probe takes
the next rung: remove the solution inside a BFS ball, solve that induced
subproblem **exactly**, and see whether the replacement is ever bigger.  A gain
here is a move of unbounded size -- exactly what the deployed loop cannot make.

    python models/probe_lns_headroom.py --test_samples 20 --budgets 1.0 \\
        --radii 2 --balls 120 --split test \\
        --test_path models/data/MIS_tensors/RB_small_author_test --save_dir outputs/v2/none_seed0
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_v2 import dataset_at  # noqa: E402
from eval_v8_capture_budget import build_capture_parser  # noqa: E402
from eval_v8_padded import shape_of  # noqa: E402
from eval_v7_budget import load_policy  # noqa: E402
from mis_lns import apply_repair, ball, neighbourhood_masks, repair  # noqa: E402
from mis_v8_pad import PaddedSearch  # noqa: E402


def popcount(value):
    return bin(value).count("1")


def candidates(selected, neighbours, balls, radius, rng):
    """BFS balls around random nodes and around the free 'ones' of the solution."""
    n = len(neighbours)
    free = [v for v in range(n) if not (selected >> v) & 1
            and neighbours[v] & selected and bin(neighbours[v] & selected).count("1") == 1]
    rng.shuffle(free)
    centres = [rng.randrange(n) for _ in range(balls)] + free[: balls // 2]
    masks = []
    seen = set()
    for centre in centres:
        region = ball(neighbours, centre, radius)
        if popcount(region) > 96 or region in seen:
            continue
        seen.add(region)
        masks.append(region)
    return masks


def probe_graph(selected, neighbours, radius, args, rng):
    regions = candidates(selected, neighbours, args.balls, radius, rng)
    gains = []
    for region in regions:
        gain, _ = repair(selected, region, neighbours)
        gains.append(gain)
    current = selected
    applied = 0
    for region, _ in sorted(zip(regions, gains), key=lambda item: -item[1]):
        gain, replacement = repair(current, region, neighbours)
        if gain > 0:
            current = apply_repair(current, region, replacement)
            applied += 1
    return dict(regions=len(regions), positive=sum(1 for gain in gains if gain > 0),
                best=max(gains) if gains else 0, applied=applied,
                greedy_gain=popcount(current) - popcount(selected))


def main():
    parser = build_capture_parser()
    parser.add_argument("--radii", type=int, nargs="+", default=[2])
    parser.add_argument("--balls", type=int, default=120, help="random ball centres per radius")
    parser.add_argument("--ball_seed", type=int, default=20260101)
    parser.add_argument("--padded_shapes", type=int, default=None)
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    output = args.result_dir / "lns"
    output.mkdir(exist_ok=True)

    solver = load_policy(args)
    solver.model.requires_grad_(False)
    device = solver.accelerator.device
    dataset = dataset_at(args.test_path, solver.dtype, args.test_samples,
                         official=args.split == "test")
    indices = list(range(len(dataset)))[args.shard::args.shards]
    n_max, capacity = shape_of(dataset, args.padded_shapes)
    engine = PaddedSearch(solver, n_max=n_max, capacity=capacity, chunk=args.chunk,
                          device=device, decode_chunk=args.decode_chunk)
    solver.num_t = 20000

    rows = []
    started = perf_counter()
    for position, index in enumerate(indices):
        graph = dataset[index]
        graph_id = f"idx_{index}" if args.split == "test" else Path(dataset.files[index]).stem
        edges = graph.edge_index
        graph = graph.to(device)
        neighbours = neighbourhood_masks(edges, int(graph.num_nodes))
        for budget in args.budgets:
            engine.load(graph, edges, args.seeds[0])
            masks = []
            engine.run(perf_counter() + budget,
                       lambda best: masks.append(engine.decode_best(best)),
                       skip_unchanged=True)
            selected = max(masks, key=int.bit_count) if masks else 0
            for radius in args.radii:
                rng = random.Random(args.ball_seed + index)
                stats = probe_graph(selected, neighbours, radius, args, rng)
                rows.append(dict(graph_id=graph_id, budget=budget, radius=radius,
                                 n=int(graph.num_nodes), size=popcount(selected), **stats))
                print(f"{position + 1}/{len(indices)} {graph_id} {budget}s r{radius} "
                      f"size {popcount(selected)} regions {stats['regions']} "
                      f"positive {stats['positive']} best +{stats['best']} "
                      f"greedy +{stats['greedy_gain']} ({stats['applied']} applied)", flush=True)

    summary = dict(ref=args.reference_seed, samples=len(indices), balls=args.balls,
                   radii=list(args.radii), budgets=list(args.budgets),
                   wall_time_s=round(perf_counter() - started, 1))
    for budget in args.budgets:
        for radius in args.radii:
            group = [row for row in rows if row["budget"] == budget and row["radius"] == radius]
            if not group:
                continue
            summary[f"{budget}s_r{radius}"] = dict(
                graphs=len(group), size=round(mean(row["size"] for row in group), 4),
                regions=round(mean(row["regions"] for row in group), 2),
                positive=round(mean(row["positive"] for row in group), 3),
                graphs_with_gain=sum(1 for row in group if row["positive"] > 0),
                best=int(max(row["best"] for row in group)),
                greedy_gain=round(mean(row["greedy_gain"] for row in group), 4))
            print(f"{budget}s r{radius}: {summary[f'{budget}s_r{radius}']}", flush=True)

    columns = list(rows[0])
    path = output / f"lns_ref{args.reference_seed}"
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[key] for key in columns])
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
