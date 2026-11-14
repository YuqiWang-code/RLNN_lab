#!/usr/bin/env python3
"""Quality as a function of the *number of search steps*.

The clock-based harness answers "what do we get in 1 s"; this one answers the
question the retraining plan needs: "does the policy reach its ceiling within
100-400 steps, or does it need thousands?"  The engine is the amortised one
(one CUDA graph for the whole split), so a 3000-step cell costs about a second
per graph.

    python models/eval_v8_steps.py --split test \
        --test_path models/data/MIS_tensors/RB_small_author_test \
        --test_samples 200 --steps 100 300 1000 3000
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

import torch

from eval_v2 import TRAIN, dataset_at, sync
from eval_v8_capture_budget import build_capture_parser
from eval_v8_padded import shape_of
from mis_v8_pad import PaddedSearch
from mis_baselines import minimum_degree
from v7_runs import backbone, load_policy


def build_parser() -> argparse.ArgumentParser:
    parser = build_capture_parser()
    parser.set_defaults(arms=["padded"], decoders=["cached"], budgets=[1.0],
                        chunk=10, decode_chunk=4, decode_every=3)
    parser.add_argument("--padded_shapes", type=int, default=None)
    parser.add_argument("--init", choices=["random", "greedy"], default="random",
                        help="prime every particle from a degree-greedy set instead of random")
    parser.add_argument("--steps", type=int, nargs="+", default=[100, 300, 1000, 3000],
                        help="search-step budgets (multiples of --chunk)")
    parser.set_defaults(result_dir=Path("outputs/v8_macro/steps"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.split == "train":
        args.test_path = str(TRAIN)
    args.save_dir = args.save_dir or str(backbone(args.reference_seed).parent)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    output = args.result_dir / "steps"
    output.mkdir(exist_ok=True)

    solver = load_policy(args)
    solver.model.requires_grad_(False)
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
    engine.load(dataset[0].to(device), dataset[0].edge_index, args.seeds[0])
    engine.run(perf_counter() + 0.05, lambda states: None)
    sync(device)

    solver.num_t = 20000
    rows = []
    started = perf_counter()
    for position, index in enumerate(indices):
        graph = dataset[index]
        graph_id = f"idx_{index}" if args.split == "test" else Path(dataset.files[index]).stem
        edges = graph.edge_index
        graph = graph.to(device)
        for seed in args.seeds:
            for budget_steps in args.steps:
                blocks = max(1, int(budget_steps) // args.chunk)
                engine.load(graph, edges, seed)
                if args.init == "greedy":
                    start = minimum_degree(graph)
                    engine.prime(sum(int(bit) << i for i, bit in enumerate(start.tolist())),
                                 n=int(graph.num_nodes))
                best = [0]

                def decode(states, box=best):
                    candidate = engine.decode_best(states)
                    if candidate.bit_count() > box[0].bit_count():
                        box[0] = candidate

                decode(engine.best_states)
                tick = perf_counter()
                steps, _, done = engine.run(float("inf"), decode, max_blocks=blocks,
                                            skip_unchanged=True,
                                            decode_every=args.decode_every)
                sync(device)
                seconds = perf_counter() - tick
                rows.append(dict(graph_id=graph_id, search_seed=seed, budget_steps=budget_steps,
                                 steps=steps, blocks=done, size=best[0].bit_count(),
                                 seconds=seconds))
                if (position + 1) % 20 == 0:
                    mean = sum(r["size"] for r in rows if r["budget_steps"] == budget_steps) / max(
                        1, sum(1 for r in rows if r["budget_steps"] == budget_steps))
                    print(f"{position + 1}/{len(indices)} {budget_steps} steps -> {mean:.4f}", flush=True)

    tag = "" if args.shards == 1 else f"_shard{args.shard}"
    path = output / f"steps_ref{args.reference_seed}{tag}.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        columns = ["graph_id", "search_seed", "budget_steps", "steps", "blocks", "size", "seconds"]
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[name] for name in columns])

    print(f"capture {capture_seconds * 1000:.1f} ms, {len(rows)} rows, "
          f"{perf_counter() - started:.1f} s -> {path}", flush=True)
    for budget_steps in args.steps:
        cell = [row for row in rows if row["budget_steps"] == budget_steps]
        if not cell:
            continue
        print(f"{budget_steps:5d} steps size {sum(row['size'] for row in cell) / len(cell):.4f} "
              f"({sum(row['seconds'] for row in cell) / len(cell):.2f} s/graph)", flush=True)


if __name__ == "__main__":
    main()
