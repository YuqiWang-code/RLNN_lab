"""Isolated paired P0 timing, with graph preparation inside the clock."""

import argparse
import csv
from pathlib import Path
from statistics import mean, median, quantiles
from time import perf_counter

import torch

from eval_v2 import TRAIN, dataset_at, search, sync, write_run
from mis_v3_search import SearchGraph, platform_search
from mis_v3_value import load_pave_v3
from v7_runs import pave, preflight_policy, provenance


def budget_probe(args, policies, ranker, dataset, device):
    rows = []
    for i, graph in enumerate(dataset):
        gid = Path(dataset.files[i]).stem
        seed = 1017 + i
        edges = graph.edge_index
        graph = graph.to(device)
        for budget in (.5, 1.):
            names = list(policies) if i % 2 == 0 else list(reversed(policies))
            for name in names:
                policy = policies[name]
                policy.num_t = 10000
                sync(device)
                tick = perf_counter()
                deadline = tick + budget
                legal = SearchGraph(graph.num_nodes, edges, seed=seed)
                selected, steps = search(policy, graph, legal, seed, deadline=deadline-.05)
                fallback = not selected
                if fallback:
                    selected = 1 << legal.greedy_order[0]
                result = platform_search(legal, selected, choose=ranker.choose, horizon=1, deadline=deadline)
                elapsed = perf_counter()-tick
                assert legal.feasible(result.best)
                rows.append(dict(graph_id=gid, arm=name, budget=budget, search_steps=steps,
                                 solve_s=elapsed, overrun=int(elapsed > budget), fallback=int(fallback)))
    return rows


@torch.no_grad()
def profile(args):
    device = torch.device("cuda")
    options = dict(ref=args.ref, width=args.width, d_state=args.d_state, cuda_graph=True)
    policies = {"baseline": preflight_policy("none", args.precision, **options),
                "mamba": preflight_policy("v7_mamba", args.precision, checkpoint=args.checkpoint, **options)}
    ranker = load_pave_v3(pave(args.ref))
    dataset = dataset_at(TRAIN, policies["baseline"].dtype, args.samples)
    for policy in policies.values():
        graph = dataset[0]
        legal = SearchGraph(graph.num_nodes, graph.edge_index, seed=1017)
        search(policy, graph.to(device), legal, 1017)
    if args.budget_probe:
        rows = budget_probe(args, policies, ranker, dataset, device)
        args.output.mkdir(parents=True, exist_ok=False)
        with (args.output / "per_graph.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        results = {}
        for budget in (.5, 1.):
            results[str(budget)] = {}
            for name in policies:
                subset = [r for r in rows if r["arm"] == name and r["budget"] == budget]
                results[str(budget)][name] = {key: mean(r[key] for r in subset)
                                               for key in ("search_steps", "solve_s", "overrun", "fallback")}
        write_run(args.output, dict(**provenance(args.ref), status="completed", split="train",
                                   width=args.width, precision=args.precision, results=results))
        return
    rows = []
    for i, graph in enumerate(dataset):
        gid = Path(dataset.files[i]).stem
        seed = 1017 + i
        edges = graph.edge_index
        graph = graph.to(device)
        for rep in range(args.repeats):
            names = list(policies) if (i + rep) % 2 == 0 else list(reversed(policies))
            for name in names:
                policy = policies[name]
                times = {}
                policy.context_timings = dict(static_features_s=0., ordering_s=0., context_s=0.)

                def observe(step, *_):
                    if step in (0, 100):
                        sync(device)
                        times[step] = perf_counter()

                sync(device)
                tick = perf_counter()
                legal = SearchGraph(graph.num_nodes, edges, seed=seed)
                selected, steps = search(policy, graph, legal, seed, observe=observe)
                sync(device)
                decoded = perf_counter()
                improved = platform_search(legal, selected, choose=ranker.choose, horizon=1).best
                final = perf_counter()
                if not legal.feasible(selected) or not legal.feasible(improved):
                    raise ValueError(f"Infeasible solution for {gid}")
                rows.append(dict(graph_id=gid, arm=name, repeat=rep, search_steps=steps,
                                 base_size=selected.bit_count(), pave_size=improved.bit_count(),
                                 prepare_capture_s=times[0]-tick, rollout_s=times[100]-times[0],
                                 a3_s=decoded-times[100], pave_s=final-decoded,
                                 bare_s=decoded-tick, full_s=final-tick, **policy.context_timings))
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "per_graph.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paired = {}
    for name in policies:
        subset = [row for row in rows if row["arm"] == name]
        keys = ("bare_s", "full_s", "static_features_s", "ordering_s", "context_s",
                "prepare_capture_s", "rollout_s", "a3_s", "pave_s", "search_steps")
        paired[name] = {key: mean(row[key] for row in subset) for key in keys}
        times = [row["bare_s"] for row in subset]
        paired[name].update(p50_s=median(times), p95_s=quantiles(times, n=100, method="inclusive")[94])
    ratio = paired["mamba"]["bare_s"] / paired["baseline"]["bare_s"]
    record = dict(**provenance(args.ref), status="completed", split="train", precision=args.precision,
                  ref=args.ref, width=args.width, d_state=args.d_state, samples=len(dataset), repeats=args.repeats,
                  checkpoint=args.checkpoint, cuda_graph=True, device=torch.cuda.get_device_name(device),
                  ratio=ratio, pass_speed=ratio <= 1.08, timings=paired)
    write_run(args.output, record)
    print(f"{args.precision} width={args.width}: {ratio:.4f}x", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("no", "fp16"), required=True)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--ref", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--budget-probe", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    profile(parser.parse_args())
