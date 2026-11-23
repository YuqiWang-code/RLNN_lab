"""Measure resident OnlineMIS from host edge indices to materialized incumbents."""
import argparse
import csv
from ctypes import CDLL, c_double, c_int, c_uint
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix
import torch

from eval_priority_search import Emit, IntPtr


def solve(raw, seed, native, budget_ms, reserve_ms):
    start = perf_counter()
    n = int(raw["num_nodes"])
    edges = raw["edge_index"].numpy()
    csr = coo_matrix((np.ones(edges.shape[1], np.int8), (edges[0], edges[1])),
                     shape=(n, n)).tocsr()
    rows, cols = csr.indptr.ctypes.data_as(IntPtr), csr.indices.ctypes.data_as(IntPtr)
    events = []

    @Emit
    def emit(size, pointer):
        vertices = np.ctypeslib.as_array(pointer, shape=(size,)).copy()
        ready = (perf_counter() - start) * 1000
        if ready <= budget_ms:
            events.append((ready, vertices))

    remaining = (budget_ms - reserve_ms) / 1000 - (perf_counter() - start)
    native(n, rows, cols, seed, max(0., remaining), emit)
    wall = (perf_counter() - start) * 1000
    for ready, vertices in events:
        chosen = np.zeros(n, dtype=bool)
        chosen[vertices] = True
        if chosen.sum() != len(vertices) or np.any(chosen[edges[0]] & chosen[edges[1]]):
            raise ValueError(f"Invalid OnlineMIS witness for seed {seed} at {ready} ms")
    return wall, events


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", required=True, type=Path)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--budget-ms", type=float, required=True)
    p.add_argument("--reserve-ms", type=float, default=3)
    p.add_argument("--seeds", type=int, nargs="+", default=[1017, 1019, 1021])
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    entries = json.loads((args.dataset / "tensor_manifest.json").read_text())["graphs"][:args.limit]
    torch.set_num_threads(1)
    lib = CDLL(str(args.library.resolve()))
    native = lib.run_online_mis
    native.argtypes = [c_int, IntPtr, IntPtr, c_uint, c_double, Emit]
    native.restype = None
    args.output.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(method="KaMIS OnlineMIS (resident)", revision="2e4b3861b8063f05e8520434b97ce59e21ca49e7",
                  graphs=len(entries), host=platform.node(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                  timing="Resident host edge_index to copied host witnesses, including CSR and KaMIS graph construction.")
    (args.output / "run.json").write_text(json.dumps(config, indent=2) + "\n")
    fields = ("graph_id", "seed", "repeat", "budget", "final_size", "wall_ms", "first_ready_ms",
              "final_ready_ms", "has_solution", "deadline_overrun")
    with (args.output / "per_graph.csv").open("w", newline="") as stream, \
         (args.output / "solutions.jsonl").open("w") as archive:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, entry in enumerate(entries):
            raw = torch.load(args.dataset / entry["tensor"], weights_only=True, map_location="cpu")
            graph_id = Path(entry["source"]).stem
            for seed in args.seeds:
                wall, events = solve(raw, seed, native, args.budget_ms, args.reserve_ms)
                row = dict(graph_id=graph_id, seed=seed, repeat=0, budget=args.budget_ms/1000,
                           final_size=len(events[-1][1]) if events else 0, wall_ms=wall,
                           first_ready_ms=events[0][0] if events else None,
                           final_ready_ms=events[-1][0] if events else None,
                           has_solution=int(bool(events)), deadline_overrun=int(wall > args.budget_ms))
                writer.writerow(row)
                archive.write(json.dumps(dict(graph_id=graph_id, seed=seed, updates=[
                    dict(ready_ms=ready, nodes=nodes.tolist()) for ready, nodes in events
                ]), separators=(",", ":")) + "\n")
            if (i+1)%20 == 0 or i+1 == len(entries):
                stream.flush()
                archive.flush()
                print(f"{i+1}/{len(entries)} graphs", flush=True)


if __name__ == "__main__":
    main()
