#!/usr/bin/env python3
"""How much headroom is left on the *deployed* solution?

The roadmap's second step is a learned exchange head.  Before spending days on
training it we need to know whether there is anything to learn: the PAVE repair
already enumerates every 1->2 and 2->3 exchange exactly (``mis_v2_exchange._exchanges``
with ``size`` 1 and 2), and it fires zero times on converged solutions.  So the
only room left is **larger** moves: k -> k+1 for k >= 3, which no component of
the current pipeline ever enumerates.

This probe measures that headroom on the 1.0 s / 0.2 s deployed answers:

* ``(1,2)`` moves are counted exactly (that is PAVE's own neighbourhood);
* ``k -> k+1`` for k = 3, 4 is estimated by sampling: pick k selected nodes at
  random, collect every free node whose whole neighbourhood inside the solution
  is contained in that sample, then greedily pack an independent set out of
  them.  If the packed set has more than k nodes, the move is real.

A move found this way is a *witness*, not a proof of optimality, so the number
reported is "fraction of graphs where a sampled move of this radius exists".
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from eval_v2 import dataset_at, load_policy, sync
from eval_v7_budget import run_pave
from eval_v8_capture_budget import build_capture_parser, captured_search
from mis_v1 import adjacency
from mis_v3_search import SearchGraph
from mis_v3_value import PAVEV3
from v7_runs import backbone, pave


def bits(mask):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def exact_ones(selected, neighbours, n):
    """Count free nodes blocked by exactly one selected node (PAVE's (1,2) set)."""
    count = 0
    for u in range(n):
        if selected >> u & 1:
            continue
        blockers = neighbours[u] & selected
        if blockers and blockers.bit_count() == 1:
            count += 1
    return count


def targeted_moves(selected, neighbours, n, k):
    """Exact k -> k+1 witnesses: candidate blocker sets are unions of node blockers.

    A node u can join only when *all* of its selected neighbours are released, so
    the released set R has to contain blockers(u).  Every candidate R with
    |R| <= k is therefore a union of the blockers of at most k free nodes --
    which is a small set, so we can enumerate it instead of sampling.
    """
    groups = {}
    for u in range(n):
        if selected >> u & 1:
            continue
        blockers = neighbours[u] & selected
        count = blockers.bit_count()
        if 0 < count <= k:
            groups.setdefault(blockers, []).append(u)
    if not groups:
        return 0, 0
    masks = list(groups)
    candidates = set(masks)
    for index, first in enumerate(masks):
        for second in masks[index + 1:]:
            union = first | second
            if union.bit_count() <= k:
                candidates.add(union)
    # a k-set R need not be the union of two blocker masks: grow each candidate
    # by single selected nodes so that genuine k-radius moves are not missed.
    members = list(bits(selected))
    grown = set()
    for union in candidates:
        for v in members:
            bigger = union | (1 << v)
            if bigger.bit_count() <= k:
                grown.add(bigger)
    candidates |= grown
    best, found = 0, 0
    for released in candidates:
        pool = [u for mask, us in groups.items() if mask & ~released == 0 for u in us]
        if len(pool) <= released.bit_count():
            continue
        pool.sort(key=lambda u: (neighbours[u] & selected).bit_count())
        picked = []
        for u in pool:
            if all(not (neighbours[u] >> v & 1) for v in picked):
                picked.append(u)
                if len(picked) > released.bit_count():
                    break
        size = len(picked)
        if size > released.bit_count():
            found += 1
            best = max(best, size)
    return best, found


def build_parser() -> argparse.ArgumentParser:
    parser = build_capture_parser()
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--radii", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--control", action="store_true",
                        help="replace the search with a greedy start (positive control)")
    parser.set_defaults(result_dir=Path("outputs/v8_macro/exchange_headroom"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.save_dir = args.save_dir or str(backbone(args.reference_seed).parent)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    output = args.result_dir / "headroom"
    output.mkdir(parents=True, exist_ok=True)

    solver = load_policy(args)
    solver.model.requires_grad_(False)
    device = solver.accelerator.device
    checkpoint = torch.load(pave(args.reference_seed), map_location="cpu", weights_only=False)
    ranker = PAVEV3(checkpoint["mode"], checkpoint.get("stats_dim", 24))
    ranker.load_state_dict(checkpoint["model_state"])
    ranker = ranker.cpu().eval()
    dataset = dataset_at(args.test_path, solver.dtype, args.test_samples,
                         official=args.split == "test")
    rng = random.Random(args.seeds[0])

    rows = []
    for graph_id, graph in enumerate(dataset):
        edges = graph.edge_index
        adj = adjacency(graph.to(device))
        topology = SearchGraph(graph.num_nodes, edges, seed=args.seeds[0])
        normalized = solver.policy_topology(graph, adj)
        neighbours = topology.neighbors
        n = graph.num_nodes
        for budget in args.budgets:
            seed = args.seeds[0]
            incumbent, masks, steps, _, prelude = captured_search(
                solver, graph, adj, seed, perf_counter() + budget,
                normalized=normalized, skip_unchanged=True)
            selected = max(masks, key=int.bit_count) if masks else 0
            deadline = perf_counter() + budget
            final = run_pave(topology, selected, ranker, args, deadline)
            if args.control:
                from mis_baselines import minimum_degree
                final = minimum_degree(graph.to(device))
                final = sum(1 << i for i in range(n) if bool(final[i]))
            sync(device)
            row = dict(graph_id=graph_id, budget=budget, nodes=n,
                       search_size=selected.bit_count(), pave_size=final.bit_count(),
                       steps=steps, prelude_s=round(prelude, 4),
                       ones=exact_ones(final, neighbours, n))
            for k in args.radii:
                best, found = targeted_moves(final, neighbours, n, k)

                row[f"moves_{k}"] = found
            rows.append(row)
            print(f"{graph_id:4d} {budget:.2f}s n{n} pave {row['pave_size']} "
                  f"ones {row['ones']} " +
                  " ".join(f"k{k}:{row[f'moves_{k}']}" for k in args.radii),
                  flush=True)

    columns = list(rows[0])
    with (output / f"headroom_ref{args.reference_seed}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print("\nsummary")
    for budget in args.budgets:
        block = [row for row in rows if row["budget"] == budget]
        line = f"{budget:.2f}s graphs {len(block)} pave {mean(r['pave_size'] for r in block):.4f} " \
               f"ones/exact {mean(r['ones'] for r in block):.2f}"
        for k in args.radii:
            graphs = sum(1 for r in block if r[f"moves_{k}"] > 0)
            line += f" | k{k}: {graphs}/{len(block)} graphs"
        print(line)


if __name__ == "__main__":
    main()
