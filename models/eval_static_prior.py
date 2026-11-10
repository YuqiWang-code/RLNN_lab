"""Interleaved MIS comparison of static action priors on one exclusive GPU."""

import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter

import torch

from eval_v2 import VALID, dataset_at, search, sync, write_run
from mis_static_prior import STATIC_FEATURES
from mis_v3_search import SearchGraph, nodes, platform_search
from mis_v3_value import load_pave_v3
from v7_runs import backbone, digest, pave, preflight_policy, provenance


ARMS = ("baseline", "constants", "centered", "masked")
PROTOCOLS = {"t100": (100, None, None), "t1000": (1000, None, 10),
             "budget05": (10000, .5, 10), "budget10": (10000, 1., 10)}


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate(args):
    steps, budget, interval = PROTOCOLS[args.protocol]
    repeats = args.repeats if budget else 1
    paths = {"baseline": backbone(args.ref)}
    for arm in ("constants", "centered"):
        paths[arm] = args.root / f"train_{arm}_ref{args.ref}/checkpoints/best.pt"
    paths["masked"] = paths["centered"]
    policies, params = {}, {}
    for arm in args.arms:
        source_arm = "centered" if arm == "masked" else arm
        if source_arm not in policies:
            mode = "none" if arm == "baseline" else f"static_{source_arm}"
            policies[source_arm] = preflight_policy(mode, "no", ref=args.ref, checkpoint=paths[arm].parent,
                                                    cuda_graph=True)
        policies[arm] = policies[source_arm]
        params[arm] = sum(p.numel() for p in policies[arm].model.parameters())
    device = next(iter(policies.values())).accelerator.device
    ranker = load_pave_v3(pave(args.ref))
    pave_params = sum(p.numel() for p in ranker.parameters())
    dataset = dataset_at(VALID, torch.float32)
    dataset.files = dataset.files[args.offset:args.offset+args.samples]
    if len(dataset) != args.samples:
        raise ValueError("The requested validation slice is incomplete")
    graph_ids = [Path(file).stem for file in dataset.files]
    args.output.mkdir(parents=True, exist_ok=False)

    def policy_for(arm, limit):
        policy = policies[arm]
        policy.num_t = limit
        if arm in ("centered", "masked"):
            policy.accelerator.unwrap_model(policy.model).adapter.centered = arm == "centered"
        return policy

    warm = dataset[0]
    legal = SearchGraph(warm.num_nodes, warm.edge_index, seed=1017+args.offset)
    warm = warm.to(device)
    for arm in args.arms:
        search(policy_for(arm, 100), warm, legal, 1017+args.offset)
    sync(device)
    started = datetime.now(timezone.utc).isoformat()
    records = []
    for repeat in range(repeats):
        current = []
        for index, graph in enumerate(dataset):
            seed = 1017 + args.offset + index
            edges = graph.edge_index
            graph = graph.to(device)
            rotation = index % len(args.arms)
            order = args.arms[rotation:] + args.arms[:rotation]
            if repeat % 2:
                order = list(reversed(order))
            for position, arm in enumerate(order):
                solver = policy_for(arm, steps)
                sync(device)
                resident = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                tick = perf_counter()
                deadline = tick + budget if budget else None
                legal = SearchGraph(graph.num_nodes, edges, seed=seed)
                timings = {}
                chosen, used = search(solver, graph, legal, seed, deadline=deadline-.05 if budget else None,
                                      decode_every=interval, timings=timings)
                sync(device)
                end_search = perf_counter()
                fallback = chosen == 0 and graph.num_nodes > 0
                if fallback:
                    chosen = 1 << legal.greedy_order[0]
                improved = platform_search(legal, chosen, choose=ranker.choose, horizon=1,
                                           deadline=deadline if budget else float("inf")).best
                if not legal.feasible(chosen) or not legal.feasible(improved):
                    raise ValueError(f"Infeasible solution on {graph_ids[index]} for {arm}")
                end = perf_counter()
                peak = torch.cuda.max_memory_allocated(device)
                current.append(dict(graph_id=graph_ids[index], search_seed=seed, repeat=repeat, order=position,
                                    arm=arm, base_size=chosen.bit_count(), final_size=improved.bit_count(),
                                    search_steps=used, search_s=end_search-tick, pave_s=end-end_search,
                                    solve_s=end-tick, **timings, feasible=1, fallback=int(fallback),
                                    overrun=int(budget is not None and end > deadline),
                                    peak_memory_mib=peak/2**20, solve_peak_mib=(peak-resident)/2**20,
                                    selected_nodes=" ".join(map(str, nodes(improved)))))
            if (index+1) % 10 == 0 or index+1 == len(dataset):
                summary = ", ".join(f"{arm}={mean(r['final_size'] for r in current if r['arm']==arm):.3f}"
                                    for arm in args.arms)
                print(f"{args.protocol} repeat {repeat} {index+1}/{len(dataset)} {summary}", flush=True)
        write_rows(args.output / f"repeat_{repeat}.csv", current)
        records.extend(current)
    write_rows(args.output / "per_graph.csv", records)
    source = provenance(args.ref)
    source["static_feature_schema"] = STATIC_FEATURES
    write_run(args.output, dict(**source, version="static_prior_v1", status="completed",
                                protocol=args.protocol, num_t=steps, budget_s=budget, decode_every=interval,
                                precision="fp32", cuda_graph=True, num_k=20, reserve_s=.05 if budget else None,
                                repeats=repeats, graph_ids=graph_ids, ref=args.ref, offset=args.offset,
                                gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                                device=torch.cuda.get_device_name(device), params=params, pave_params=pave_params,
                                checkpoint_sha256={arm: digest(paths[arm]) for arm in args.arms}, started_at=started,
                                finished_at=datetime.now(timezone.utc).isoformat()))


if __name__ == "__main__":
    if sys.argv[1:2] == ["v2"]:
        from eval_static_prior_v2 import main
        del sys.argv[1]
        main()
        sys.exit()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, choices=PROTOCOLS)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ref", type=int, default=0)
    parser.add_argument("--offset", type=int, default=50)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    args = parser.parse_args()
    if args.offset < 0 or min(args.samples, args.repeats) < 1 or len(args.arms) != len(set(args.arms)):
        parser.error("Use a nonnegative offset, positive sample/repeat counts and distinct arms")
    evaluate(args)
