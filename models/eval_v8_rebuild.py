"""Destroy-and-rebuild with the frozen policy as the rebuilder.

Clearing a region with a *greedy* refill cannot beat the incumbent, because the
greedy is weaker than the network that produced it.  This arm clears the closed
neighbourhood of a connected cluster of selected points and then lets the frozen
policy re-converge inside that hole: ``prime`` writes the cleared solution into
every particle, the captured loop runs a fixed number of steps, and the decoded
result replaces the incumbent if it is larger.  Repeat until the clock runs out.

    python models/eval_v8_rebuild.py --split test \
        --test_path models/data/MIS_tensors/RB_small_author_test \
        --test_samples 100 --budgets 1.0 --arms search rebuild \
        --save_dir outputs/v2/none_seed0/checkpoints --reference_seed 0
"""

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_v2 import dataset_at  # noqa: E402
from eval_v7_budget import load_policy  # noqa: E402
from eval_v8_capture_budget import build_capture_parser  # noqa: E402
from eval_v8_padded import shape_of  # noqa: E402
from mis_lns import ball, neighbourhood_masks  # noqa: E402
from mis_v8_pad import PaddedSearch  # noqa: E402


def build_parser():
    parser = build_capture_parser()
    for action in parser._actions:
        if action.dest == "arms":
            action.choices = None
            action.default = ["search", "rebuild"]
    parser.add_argument("--reserve", type=float, default=0.25,
                        help="fraction of the budget kept for the rebuild cycles")
    parser.add_argument("--cycle_steps", type=int, default=60)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--max_cycles", type=int, default=0)
    parser.add_argument("--ball_seed", type=int, default=20260101)
    parser.add_argument("--padded_shapes", type=int, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    output = args.result_dir / "rebuild"
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
    print(f"shapes n_max {n_max} capacity {capacity} graphs {len(indices)}", flush=True)

    rows = []
    started = perf_counter()
    for position, index in enumerate(indices):
        graph = dataset[index]
        graph_id = f"idx_{index}" if args.split == "test" else Path(dataset.files[index]).stem
        edges = graph.edge_index
        graph = graph.to(device)
        nodes = int(graph.num_nodes)
        neighbours = neighbourhood_masks(edges, nodes)
        for budget in args.budgets:
            for arm in args.arms:
                begin = perf_counter()
                deadline = begin + budget
                load_started = perf_counter()
                engine.load(graph, edges, args.seeds[0])
                load_s = perf_counter() - load_started
                buckets = []
                if arm == "search":
                    engine.run(deadline, lambda best: buckets.append(engine.decode_best(best)),
                               skip_unchanged=True)
                    incumbent = max(buckets, key=int.bit_count) if buckets else 0
                    cycles = positive = 0
                    refill_total = refill_best = 0
                    removed_total = cluster_total = 0
                else:
                    cutoff = begin + budget * (1.0 - args.reserve)
                    engine.run(cutoff, lambda best: buckets.append(engine.decode_best(best)),
                               skip_unchanged=True)
                    incumbent = max(buckets, key=int.bit_count) if buckets else 0
                    rng = random.Random(args.ball_seed + index)
                    cycles = positive = 0
                    refill_total = refill_best = 0
                    removed_total = cluster_total = 0
                    while perf_counter() < deadline:
                        if args.max_cycles and cycles >= args.max_cycles:
                            break
                        points = [v for v in range(nodes) if (incumbent >> v) & 1]
                        if not points:
                            break
                        region = ball(neighbours, rng.choice(points), args.radius)
                        candidate = incumbent & ~region
                        removed_total += (incumbent & region).bit_count()
                        cluster_total += region.bit_count()
                        engine.prime(candidate, n=nodes)
                        fresh = []
                        engine.run(min(deadline, perf_counter() + 5.0),
                                   lambda best: fresh.append(engine.decode_best(best)),
                                   max_blocks=max(1, args.cycle_steps // args.chunk),
                                   skip_unchanged=True)
                        cycles += 1
                        gained = (max(fresh, key=int.bit_count).bit_count() if fresh else 0) - candidate.bit_count()
                        refill_total += gained
                        refill_best = max(refill_best, gained)
                        if fresh:
                            trial = max(fresh, key=int.bit_count)
                            if trial.bit_count() > incumbent.bit_count():
                                incumbent = trial
                                positive += 1
                elapsed = perf_counter() - begin
                rows.append(dict(
                    graph_id=graph_id, search_seed=args.seeds[0], budget=budget, arm=arm,
                    repeat=0, nodes=nodes, final_size=incumbent.bit_count(), cycles=cycles,
                    positive_cycles=positive, refill_gain=refill_total, refill_best=refill_best,
                    removed=removed_total, cluster=cluster_total, load_s=round(load_s, 6),
                    solve_time_s=round(elapsed, 6), ms_per_step=0.0,
                    deadline_overrun=int(elapsed > budget + 0.05)))
                row = rows[-1]
                print(f"{position + 1}/{len(indices)} {graph_id} {budget}s {arm} "
                      f"size {row['final_size']} cycles {cycles} +{positive} "
                      f"wall {elapsed:.3f}s", flush=True)

    print(f"wall {perf_counter() - started:.1f}s", flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["budget"], row["arm"])].append(row)
    for key in sorted(groups):
        group = groups[key]
        print(f"{key[0]}s {key[1]}: size {sum(r['final_size'] for r in group) / len(group):.4f} "
              f"graphs {len(group)} cycles {sum(r['cycles'] for r in group) / len(group):.1f} "
              f"positive {sum(r['positive_cycles'] for r in group) / len(group):.2f}", flush=True)

    columns = list(rows[0])
    path = output / f"rebuild_ref{args.reference_seed}"
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[key] for key in columns])
    print(f"wrote {path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
