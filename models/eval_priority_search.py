"""Evaluate learned MIS priorities with a host-visible 200 ms incumbent archive."""
import argparse
from ctypes import CDLL, CFUNCTYPE, POINTER, byref, c_double, c_float, c_int32, c_uint32
import csv
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix
import torch


IntPtr = POINTER(c_int32)
FloatPtr = POINTER(c_float)
Emit = CFUNCTYPE(None, c_int32, IntPtr)


def load_native(args):
    library = CDLL(str(args.library.resolve()))
    greedy = library.priority_seed
    greedy.argtypes = [c_int32, IntPtr, IntPtr, c_uint32, IntPtr]
    greedy.restype = c_int32
    search = library.priority_search
    search.argtypes = [c_int32, IntPtr, IntPtr, FloatPtr, c_uint32, c_double,
                       c_float, c_int32, c_int32, IntPtr, c_int32, Emit]
    search.restype = c_int32
    refine = None
    if args.refine_ms:
        refine = CDLL(str(args.refiner.resolve())).clique_refine
        refine.argtypes = [c_int32, IntPtr, IntPtr, FloatPtr, c_uint32, c_double, IntPtr, c_int32, IntPtr]
        refine.restype = c_int32
    return greedy, search, refine


def load_scorer(checkpoint):
    if checkpoint is None:
        return None
    from scale_priority import PriorityScorer
    scorer = PriorityScorer.load(checkpoint, device="cuda:0")
    offsets = torch.tensor([0, 1, 3, 4], dtype=torch.int32, device="cuda:0")
    nodes = torch.tensor([1, 0, 2, 1], dtype=torch.int32, device="cuda:0")
    with torch.inference_mode():
        scorer.score(offsets, nodes).cpu()
    return scorer


def standardized(values):
    centered = values - values.mean()
    return centered / max(float(np.sqrt(np.mean(centered * centered))), 1e-4)


@torch.inference_mode()
def solve(raw, seed, args, greedy, search, refine, scorer):
    started = perf_counter()
    clock = args.budget_ms / 1000
    stop = started + (args.budget_ms - args.reserve_ms) / 1000
    n = int(raw["num_nodes"])
    edges = raw["edge_index"].numpy()
    events = []

    def archive(nodes):
        if events and len(nodes) <= len(events[-1][1]):
            return
        snapshot = nodes.copy()
        elapsed = perf_counter() - started
        if elapsed <= clock:
            events.append((elapsed * 1000, snapshot))

    csr = coo_matrix((np.ones(edges.shape[1], np.int8), (edges[0], edges[1])),
                     shape=(n, n)).tocsr()
    crow, col = csr.indptr, csr.indices
    rows = crow.ctypes.data_as(IntPtr)
    cols = col.ctypes.data_as(IntPtr)
    baseline = np.empty(n, dtype=np.int32)
    size = greedy(n, rows, cols, seed, baseline.ctypes.data_as(IntPtr))
    archive(baseline[:size])
    prepare_ms = (perf_counter() - started) * 1000
    score_start = perf_counter()
    if scorer is None:
        priority = np.zeros(n, np.float32)
    else:
        rows_gpu = torch.from_numpy(crow).cuda()
        cols_gpu = torch.from_numpy(col).cuda()
        priority = scorer.score(rows_gpu, cols_gpu).float().contiguous().cpu().numpy()
    if args.degree_prior:
        structural = standardized(-np.diff(crow).astype(np.float32))
        priority = structural if scorer is None else structural + args.learned_mix * standardized(priority)
    elif args.neural_only:
        if scorer is None:
            raise SystemExit("--neural-only needs --checkpoint")
        priority = standardized(priority)
    score_ms = (perf_counter() - score_start) * 1000
    scores = priority.ctypes.data_as(FloatPtr)

    @Emit
    def emit(count, ptr):
        archive(np.ctypeslib.as_array(ptr, shape=(count,)))

    upper = n
    def clique(seconds):
        nonlocal upper
        current = events[-1][1]
        nodes = np.empty(n, dtype=np.int32)
        nodes[:len(current)] = current
        bound = c_int32(n)
        size = refine(n, rows, cols, scores, seed, max(0, seconds),
                      nodes.ctypes.data_as(IntPtr), len(current), byref(bound))
        archive(nodes[:size])
        if perf_counter() - started <= clock:
            upper = bound.value

    def local(seconds, local_seed):
        current = events[-1][1]
        return search(n, rows, cols, scores, local_seed, max(0, seconds), args.weight,
                      args.plateau, args.perturb, current.ctypes.data_as(IntPtr), len(current), emit)

    steps = 0
    if args.refine_first and events:
        clique(min(args.refine_ms / 1000, stop - perf_counter()))
    if events and len(events[-1][1]) < upper:
        search_seconds = stop - perf_counter() - (0 if args.refine_first else args.refine_ms / 1000)
        steps += local(search_seconds, seed)
    if refine is not None and not args.refine_first and events and perf_counter() < stop:
        clique(stop - perf_counter())
        if len(events[-1][1]) < upper and stop - perf_counter() > .0005:
            steps += local(stop - perf_counter(), seed + 1)
    wall = (perf_counter() - started) * 1000
    if events and not len(events[-1][1]) <= upper <= n:
        raise ValueError(f"Upper bound {upper} contradicts incumbent for seed {seed}")
    for ready, nodes in events:
        chosen = np.zeros(n, dtype=bool)
        chosen[nodes] = True
        if chosen.sum() != len(nodes) or np.any(chosen[edges[0]] & chosen[edges[1]]):
            raise ValueError(f"Invalid incumbent for seed {seed} at {ready} ms")
    return wall, events, steps, prepare_ms, score_ms, upper


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--library", required=True, type=Path)
    p.add_argument("--refiner", type=Path)
    p.add_argument("--refine-ms", type=float, default=0)
    p.add_argument("--refine-first", action="store_true")
    p.add_argument("--budget-ms", type=float, default=200)
    p.add_argument("--reserve-ms", type=float, default=3)
    p.add_argument("--weight", type=float, default=0.7)
    p.add_argument("--degree-prior", action="store_true")
    p.add_argument("--learned-mix", type=float, default=1)
    p.add_argument("--neural-only", action="store_true")
    p.add_argument("--plateau", type=int, default=50)
    p.add_argument("--perturb", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[1017, 1019, 1021])
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    if (not 0 <= args.refine_ms < args.budget_ms - args.reserve_ms or args.perturb < 1
            or args.plateau < 1 or (args.refine_first and not args.refine_ms)):
        p.error("invalid search duration or exchange settings")
    entries = json.loads((args.dataset / "tensor_manifest.json").read_text())["graphs"]
    if args.limit:
        entries = entries[:args.limit]
    torch.set_num_threads(1)
    greedy, search, refine = load_native(args)
    scorer = load_scorer(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    run = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    run.update(graphs=len(entries), host=platform.node(), torch=torch.__version__,
               cpu_affinity=sorted(os.sched_getaffinity(0)),
               accelerator=torch.cuda.get_device_name(0) if scorer else None,
               gpu=os.environ.get("CUDA_VISIBLE_DEVICES") if scorer else None,
               timing="Resident host edge_index to materialized host incumbents. CSR construction, GPU scoring, device transfers and search are timed.")
    (args.output / "run.json").write_text(json.dumps(run, indent=2)+"\n")
    fields = ("graph_id","seed","repeat","budget","final_size","wall_ms","final_ready_ms",
              "first_ready_ms","steps","deadline_overrun","has_solution","prepare_ms","score_ms","upper_bound")
    with (args.output / "per_graph.csv").open("w", newline="") as output, \
         (args.output / "solutions.jsonl").open("w") as archive:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for i, entry in enumerate(entries):
            raw = torch.load(args.dataset / entry["tensor"], weights_only=True, map_location="cpu")
            graph_id = Path(entry["source"]).stem
            for seed in args.seeds:
                wall, events, steps, prep, score, upper = solve(raw, seed, args, greedy, search, refine, scorer)
                row = dict(graph_id=graph_id, seed=seed, repeat=0, budget=args.budget_ms/1000,
                           final_size=len(events[-1][1]) if events else 0, wall_ms=wall,
                           final_ready_ms=events[-1][0] if events else None,
                           first_ready_ms=events[0][0] if events else None, steps=steps,
                           deadline_overrun=int(wall>args.budget_ms), has_solution=int(bool(events)),
                           prepare_ms=prep, score_ms=score, upper_bound=upper)
                writer.writerow(row)
                updates = [dict(ready_ms=ready,nodes=nodes.tolist()) for ready,nodes in events]
                archive.write(json.dumps(dict(graph_id=graph_id,seed=seed,updates=updates),
                                         separators=(",",":"))+"\n")
            if (i+1)%20 == 0 or i+1 == len(entries):
                output.flush()
                archive.flush()
                print(f"{i+1}/{len(entries)} graphs",flush=True)


if __name__ == "__main__":
    main()
