"""Time Gurobi's neighborhood formulation for maximum independent set."""

import argparse
import csv
import json
import os
import platform
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import gurobipy as gp
import numpy as np
from scipy.sparse import csr_matrix, diags
import torch


SEEDS = (1017, 1019, 1021)


def solve(env, raw, seed, budget_ms):
    start = perf_counter()
    cutoff = start + budget_ms / 1000
    n = int(raw["num_nodes"])
    row, col = raw["edge_index"].numpy()
    adjacency = csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    events = []

    def publish(values):
        nodes = np.flatnonzero(np.asarray(values) > 0.5)
        t = perf_counter()
        if t <= cutoff and (not events or len(nodes) > len(events[-1][1])):
            events.append(((t - start) * 1000, nodes))

    with gp.Model(env=env) as model:
        model.Params.Threads = 1
        model.Params.Seed = seed
        x = model.addMVar(n, vtype=gp.GRB.BINARY, obj=1.0)
        model.ModelSense = gp.GRB.MAXIMIZE
        model.addMConstr(adjacency + diags(degree), x, "<", degree)
        model.update()
        variables = model.getVars()

        def capture(m, where):
            if where == gp.GRB.Callback.MIPSOL and perf_counter() <= cutoff:
                publish(m.cbGetSolution(variables))

        model.Params.TimeLimit = max(0., cutoff - perf_counter() - .003)
        model.optimize(capture)
        if model.SolCount and perf_counter() <= cutoff:
            publish(x.X)
    wall = (perf_counter() - start) * 1000
    for _, nodes in events:
        selected = np.zeros(n, dtype=bool)
        selected[nodes] = True
        if np.any(selected[row] & selected[col]):
            raise ValueError("Gurobi reported adjacent vertices in an incumbent")
    return wall, [dict(ready_ms=ms, nodes=nodes.tolist()) for ms, nodes in events]


def run_shard(dataset, entries, cpu, budget):
    os.sched_setaffinity(0, {cpu})
    torch.set_num_threads(1)
    records = []
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        solve(env, {"num_nodes": 2, "edge_index": torch.tensor([[0, 1], [1, 0]])}, 0, budget)
        for entry in entries:
            graph_id = Path(entry["source"]).stem
            raw = torch.load(dataset / entry["tensor"], weights_only=True, map_location="cpu")
            for seed in SEEDS:
                try:
                    wall, events = solve(env, raw, seed, budget)
                except gp.GurobiError as exc:
                    if exc.errno != 10010:
                        raise
                    return {"status": "License limit", "reason": str(exc), "error_code": exc.errno,
                            "first_rejected_graph": graph_id, "variables": int(raw["num_nodes"]),
                            "constraints": int(raw["num_nodes"])}
                records.append((graph_id, seed, wall, events))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cores", nargs="+", type=int, required=True)
    parser.add_argument("--budget-ms", type=int, default=200)
    args = parser.parse_args()
    entries = json.loads((args.dataset / "tensor_manifest.json").read_text())["graphs"]
    cpus = args.cores[:len(entries)]
    args.output.mkdir(parents=True, exist_ok=False)
    config = dict(method="Gurobi (neighborhood MIP)", formulation="degree[v]*x[v] + sum(x[u] for u adjacent to v) <= degree[v]",
                  graph_count=len(entries), dataset=str(args.dataset), budget_ms=args.budget_ms,
                  return_reserve_ms=3, seeds=SEEDS, threads_per_solve=1, cpu_affinity=cpus,
                  host=platform.node(), gurobi_version=list(gp.gurobi.version()),
                  timing="Resident host edge list through sparse model construction, MIPSOL host copies, optimize and disposal")
    (args.output / "run.json").write_text(json.dumps(config, indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=len(cpus), mp_context=get_context("spawn")) as pool:
        jobs = [pool.submit(run_shard, args.dataset, entries[i::len(cpus)], cpu, args.budget_ms)
                for i, cpu in enumerate(cpus)]
        shards = [job.result() for job in jobs]
    for result in shards:
        if isinstance(result, dict):
            (args.output / "status.json").write_text(json.dumps(result, indent=2) + "\n")
            print(result, flush=True)
            return
    by_graph = {Path(e["source"]).stem: i for i, e in enumerate(entries)}
    records = sorted((r for shard in shards for r in shard), key=lambda r: (by_graph[r[0]], r[1]))
    fields = ("graph_id", "seed", "repeat", "budget", "final_size", "wall_ms", "first_ready_ms",
              "final_ready_ms", "has_solution", "deadline_overrun")
    with (args.output / "per_graph.csv").open("w", newline="") as metrics, (args.output / "solutions.jsonl").open("w") as archive:
        writer = csv.DictWriter(metrics, fieldnames=fields)
        writer.writeheader()
        for graph_id, seed, wall, events in records:
            writer.writerow(dict(graph_id=graph_id, seed=seed, repeat=0, budget=args.budget_ms/1000,
                                 final_size=len(events[-1]["nodes"]) if events else 0, wall_ms=wall,
                                 first_ready_ms=events[0]["ready_ms"] if events else None,
                                 final_ready_ms=events[-1]["ready_ms"] if events else None,
                                 has_solution=int(bool(events)), deadline_overrun=int(wall>args.budget_ms)))
            archive.write(json.dumps(dict(graph_id=graph_id, seed=seed, updates=events), separators=(",", ":"))+"\n")
    print(args.dataset.name, len(records), "completed", flush=True)


if __name__ == "__main__":
    main()
