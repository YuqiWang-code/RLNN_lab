"""Replay paired flips and prepare oracle targets separately by split."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval_static_prior import write_rows
from eval_static_prior_v2 import (STEPS, Experiment, add_common, baseline, cohort, initial_state,
                                  probability_metrics, search_seeds)
from eval_v2 import TRAIN, dataset_at, search, write_run
from mis_baselines import kamis
from mis_static_prior import bias_stats
from mis_v1 import adjacency
from mis_v3_search import SearchGraph, node_mask, nodes


@torch.no_grad()
def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    if args.train_targets:
        dtype = torch.float16 if args.train_oracle == "b" and args.backend == "fp16-eager" else torch.float32
        dataset = dataset_at(TRAIN, dtype, args.samples)
        graph_ids = [Path(path).stem for path in dataset.files]
        if len(dataset) != args.samples or args.offset:
            raise ValueError("Training targets require a complete training cohort and offset 0")
    else:
        dataset, graph_ids = cohort(args)
    use_kamis = args.kamis is not None and (not args.train_targets or args.train_oracle == "kamis")
    use_base = not args.kamis_only and (not args.train_targets or args.train_oracle in ("b", "freq"))
    frequency = args.train_targets and args.train_oracle == "freq"
    base = baseline(args) if use_base else None
    if base is not None:
        base.num_t = args.b_steps
    kamis_targets, b_targets, freq_targets = {}, {}, {}
    for index, (gid, graph) in enumerate(zip(graph_ids, dataset), args.offset):
        edges = graph.edge_index
        if use_kamis:
            selected = kamis(graph, args.kamis.resolve(), 1017 + index, args.kamis_seconds).nonzero().flatten().tolist()
            if not SearchGraph(graph.num_nodes, edges, seed=1017+index).feasible(node_mask(selected)):
                raise ValueError(f"ReduMIS returned a conflicting set on {gid}")
            kamis_targets[gid] = selected
        if base is not None:
            graph = graph.to(base.accelerator.device)
            starts = args.freq_starts if frequency else args.seeds_per_graph
            solutions = []
            for seed in search_seeds(index, starts):
                seed += 1_000_003
                legal = SearchGraph(graph.num_nodes, edges, seed=seed)
                selected, _ = search(base, graph, legal, seed)
                solutions.append(selected)
            best = max(solutions, key=lambda mask: mask.bit_count())
            b_targets[gid] = list(nodes(best))
            if frequency:
                elite = sorted(solutions, key=lambda mask: -mask.bit_count())[:max(1, len(solutions) // 2)]
                counts = np.zeros(graph.num_nodes, dtype=np.float32)
                for mask in elite:
                    counts[list(nodes(mask))] += 1
                freq_targets[gid] = (counts / len(elite)).tolist()
    suffix = "-train" if args.train_targets else ""
    for name, targets in (("oracle-kamis", kamis_targets), ("oracle-b", b_targets), ("oracle-freq", freq_targets)):
        if targets:
            (args.output / f"{name}{suffix}.json").write_text(json.dumps(targets, indent=2) + "\n", encoding="utf-8")
    write_run(args.output, dict(split="train" if args.train_targets else "validation", graph_ids=graph_ids,
                                backend=args.backend, ref=args.ref, kamis_binary=args.kamis,
                                kamis_seconds=args.kamis_seconds, b_steps=args.b_steps, b_seed_offset=1_000_003,
                                freq_starts=args.freq_starts if frequency else None,
                                seeds_per_graph=args.seeds_per_graph))


@torch.no_grad()
def diagnose(args):
    dataset, graph_ids = cohort(args)
    run = Experiment(args, graph_ids)
    for policy in set(run.policies.values()) - {run.base}:
        policy.accelerator.unwrap_model(policy.model).adapter.offline_features = True
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    milestones = sorted(set(args.steps))
    for index, (gid, graph) in enumerate(zip(graph_ids, dataset), args.offset):
        legal = SearchGraph(graph.num_nodes, graph.edge_index, seed=1017+index)
        graph = graph.to(run.device)
        adj = adjacency(graph)
        topo_b = run.base.policy_topology(graph, adj)
        for seed in search_seeds(index, args.seeds_per_graph):
            initial, rng = initial_state(graph, run.base, seed)
            noise = [torch.rand(initial.shape, dtype=initial.dtype, device=initial.device, generator=rng)
                     for _ in range(milestones[-1])]
            x_base = initial
            base_at = {}
            for step, uniform in enumerate(noise, 1):
                prob = run.base.flip_logits(x_base, graph, adj, topo_b).sigmoid()
                flips = uniform < prob
                next_base = torch.where(flips, 1-x_base, x_base)
                if step in milestones:
                    base_at[step] = x_base, next_base, prob, flips
                x_base = next_base
            for arm in run.arms:
                solver = run.policy(arm, gid, index, legal, milestones[-1])
                topo = solver.policy_topology(graph, adj)
                bias = topo.cached_action_bias if solver.v7_context else initial.new_zeros((1, 2))
                stats = bias_stats(bias)
                x = initial
                for step, uniform in enumerate(noise, 1):
                    p = solver.flip_logits(x, graph, adj, topo).sigmoid()
                    flip = uniform < p
                    x = torch.where(flip, 1-x, x)
                    if step in base_at:
                        previous, base_state, prob, base_flip = base_at[step]
                        same_state = solver.flip_logits(previous, graph, adj, topo).sigmoid()
                        rows.append(dict(graph_id=gid, search_seed=seed, arm=arm, step=step, **stats,
                                         **probability_metrics(same_state, prob, uniform),
                                         trajectory_flip_disagreement=float((flip != base_flip).float().mean()),
                                         trajectory_state_disagreement=float((x != base_state).float().mean())))
    write_rows(args.output / "trajectory.csv", rows)
    write_run(args.output, dict(backend="fp32-eager" if args.backend == "fp32-graph" else args.backend,
                                paired_protocol=args.backend, steps=milestones, graph_ids=graph_ids,
                                arms=run.arms, config=vars(args)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common(parser)
    parser.add_argument("--steps", nargs="+", type=int, default=STEPS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train-targets", action="store_true")
    parser.add_argument("--train-oracle", choices=("kamis", "b", "freq"), default="kamis")
    parser.add_argument("--kamis-only", action="store_true")
    parser.add_argument("--kamis", type=Path)
    parser.add_argument("--kamis-seconds", type=float, default=1.)
    parser.add_argument("--b-steps", type=int, default=800)
    parser.add_argument("--freq-starts", type=int, default=4)
    parser.set_defaults(samples=None)
    args = parser.parse_args()
    if args.samples is None:
        args.samples = 1000 if args.train_targets else 100
    if args.offset < 0 or min(args.samples, args.seeds_per_graph, *args.steps) < 1:
        parser.error("Use a nonnegative offset and positive graph/seed/step counts")
    if args.kamis_only and (not args.prepare or args.kamis is None or args.train_targets):
        parser.error("--kamis-only requires --prepare and --kamis on validation")
    if args.train_targets and (not args.prepare or (args.train_oracle == "kamis" and args.kamis is None)):
        parser.error("--train-targets requires --prepare and --kamis when using KaMIS labels")
    (prepare if args.prepare else diagnose)(args)


if __name__ == "__main__":
    main()
