"""Destroy-and-rebuild: clear the closed neighbourhood of a connected cluster.

The exact ball repair is powerless because a dense region leaves almost nothing
admissible once points outside the region block it.  This operator sidesteps the
boundary: pick a *connected cluster* of selected vertices, clear every selected
point around it, and refill the cleared area greedily.  The refill is heuristic
on purpose -- an exact solve over a closed region on an RB graph (average degree
38) would be several hundred nodes.

    python models/probe_cluster_rebuild.py --test_samples 20 --budgets 1.0 \
        --sizes 8 16 --rounds 20 --split test \
        --test_path models/data/MIS_tensors/RB_small_author_test
"""

import random
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_v2 import dataset_at  # noqa: E402
from eval_v7_budget import load_policy  # noqa: E402
from eval_v8_capture_budget import build_capture_parser  # noqa: E402
from eval_v8_padded import shape_of  # noqa: E402
from mis_lns import neighbourhood_masks  # noqa: E402
from mis_v8_pad import PaddedSearch  # noqa: E402


def popcount(value):
    return bin(value).count("1")


def cluster_of(selected, neighbours, size, rng):
    """Grow a connected cluster of selected vertices from a random seed."""
    points = [v for v in range(len(neighbours)) if (selected >> v) & 1]
    if not points:
        return 0
    cluster = 1 << rng.choice(points)
    while popcount(cluster) < size:
        frontier = 0
        rest = cluster
        while rest:
            bit = rest & -rest
            rest ^= bit
            frontier |= neighbours[bit.bit_length() - 1] & selected
        frontier &= ~cluster
        if not frontier:
            break
        options = [v for v in range(len(neighbours)) if (frontier >> v) & 1]
        cluster |= 1 << rng.choice(options)
    return cluster


def closed_region(cluster, neighbours):
    """The cluster plus every neighbour of it (all points that touch it)."""
    region = cluster
    rest = cluster
    while rest:
        bit = rest & -rest
        rest ^= bit
        region |= neighbours[bit.bit_length() - 1]
    return region


def greedy_mis(mask, neighbours):
    """Maximal independent set of ``mask``, cheapest-degree-first."""
    degrees = []
    rest = mask
    while rest:
        bit = rest & -rest
        rest ^= bit
        vertex = bit.bit_length() - 1
        degrees.append((bin(neighbours[vertex] & mask).count("1"), vertex))
    degrees.sort()
    chosen = 0
    for _, vertex in degrees:
        if not (chosen >> vertex) & 1 and not (neighbours[vertex] & chosen):
            chosen |= 1 << vertex
    return chosen


def rebuild(selected, neighbours, size, rng):
    """One destroy-and-rebuild round; returns the new selection."""
    cluster = cluster_of(selected, neighbours, size, rng)
    if not cluster:
        return selected
    region = closed_region(cluster, neighbours)
    kept = selected & ~region
    blocked = 0
    rest = kept
    while rest:
        bit = rest & -rest
        rest ^= bit
        blocked |= neighbours[bit.bit_length() - 1]
    allowed = region & ~blocked
    return kept | greedy_mis(allowed, neighbours)


def main():
    parser = build_capture_parser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--ball_seed", type=int, default=20260101)
    parser.add_argument("--padded_shapes", type=int, default=None)
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    output = args.result_dir / "cluster"
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
                       lambda best: masks.append(engine.decode_best(best)), skip_unchanged=True)
            selected = max(masks, key=int.bit_count) if masks else 0
            base = popcount(selected)
            for size in args.sizes:
                rng = random.Random(args.ball_seed + index + size)
                current = selected
                gains = []
                for _ in range(args.rounds):
                    proposal = rebuild(current, neighbours, size, rng)
                    gains.append(popcount(proposal) - popcount(current))
                    current = proposal
                rows.append(dict(graph_id=graph_id, budget=budget, size=size, n=int(graph.num_nodes),
                                 base=base, final=popcount(current), gain=popcount(current) - base,
                                 improved=int(popcount(current) > base), rounds=args.rounds,
                                 positive_rounds=sum(1 for gain in gains if gain > 0),
                                 best_round=max(gains) if gains else 0))
                last = rows[-1]
                print(f"{position + 1}/{len(indices)} {graph_id} {budget}s c{size} "
                      f"base {base} final {last['final']} gain {last['gain']:+d} "
                      f"positive rounds {last['positive_rounds']}", flush=True)

    print(f"wall {perf_counter() - started:.1f}s", flush=True)
    for budget in args.budgets:
        for size in args.sizes:
            group = [row for row in rows if row["budget"] == budget and row["size"] == size]
            if not group:
                continue
            print(f"{budget}s c{size}: graphs {len(group)} base {mean(r['base'] for r in group):.3f} "
                  f"final {mean(r['final'] for r in group):.3f} "
                  f"gain {mean(r['gain'] for r in group):+.3f} "
                  f"improved {sum(r['improved'] for r in group)}/{len(group)} "
                  f"positive rounds {mean(r['positive_rounds'] for r in group):.2f}", flush=True)

    import csv
    columns = list(rows[0])
    path = output / f"cluster_ref{args.reference_seed}"
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[key] for key in columns])
    print(f"wrote {path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
