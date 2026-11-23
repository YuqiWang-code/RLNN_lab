"""Independently bound the MIS size of a completed RB test run with SAT."""
import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import torch
from pysat import __version__ as pysat_version
from pysat.card import CardEnc, EncType
from pysat.solvers import Solver


def clique_cover(closed, edges):
    members = [1 << i for i in range(len(closed))]
    owner = list(range(len(closed)))
    shared = closed.copy()
    pairs = sorted(edges, key=lambda uv: (closed[uv[0]] & closed[uv[1]]).bit_count()
                   / (closed[uv[0]] | closed[uv[1]]).bit_count(), reverse=True)
    for u, v in pairs:
        a, b = owner[u], owner[v]
        if a == b or members[b] & ~shared[a]:
            continue
        bits = members[b]
        while bits:
            bit = bits & -bits
            owner[bit.bit_length()-1] = a
            bits ^= bit
        members[a] |= members[b]
        members[b] = 0
        shared[a] &= shared[b]
    clusters = [[i for i in range(len(closed)) if mask >> i & 1] for mask in members if mask]
    assert sorted(v for cluster in clusters for v in cluster) == list(range(len(closed)))
    assert all(all(closed[u] >> v & 1 for u in cluster for v in cluster) for cluster in clusters)
    return clusters


def verify(case):
    graph_id, tensor, delivered = case
    start = perf_counter()
    graph = torch.load(tensor, map_location="cpu", weights_only=True)
    n = int(graph["num_nodes"])
    edges = [(int(u), int(v)) for u,v in graph["edge_index"].T.numpy() if u<v]
    closed = [1 << i for i in range(n)]
    hard = []
    for u, v in edges:
        closed[u] |= 1 << v
        closed[v] |= 1 << u
        hard.append([-(u+1), -(v+1)])
    cover = clique_cover(closed, edges)
    if delivered > len(cover):
        raise ValueError(f"Feasible size exceeds verified clique bound for {graph_id}")
    skips = []
    for j, group in enumerate(cover):
        skip = n+j+1
        skips.append(skip)
        hard.append([skip, *(v+1 for v in group)])
    chosen = delivered
    witness = []
    while chosen < len(cover):
        # At most one vertex can be selected per verified clique. Limiting
        # empty cliques is therefore equivalent to asking for size >= chosen+1.
        at_most = CardEnc.atmost(skips, bound=len(skips)-chosen-1,
                                 top_id=n+len(skips), encoding=EncType.seqcounter)
        with Solver(name="cadical195", bootstrap_with=hard+at_most.clauses) as solver:
            if not solver.solve():
                break
            feasible = [lit-1 for lit in solver.get_model() if 1 <= lit <= n]
        mask = sum(1<<v for v in feasible)
        assert len(feasible)>chosen and all((mask & closed[v]) == 1<<v for v in feasible)
        chosen = len(feasible)
        witness = feasible
    return dict(graph_id=graph_id, best_delivered=delivered, optimum=chosen,
                clique_upper=len(cover), elapsed_s=round(perf_counter()-start,4),
                larger_witness=json.dumps(witness,separators=(",",":")))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers",type=int,default=1)
    p.add_argument("--indices",type=int,nargs="+",help="restrict to the specified author graph indices")
    args = p.parse_args()
    torch.set_num_threads(1)
    with (args.run/"per_graph.csv").open(newline="") as f:
        sizes = {}
        for row in csv.DictReader(f):
            key = row["graph_id"]
            sizes[key] = max(sizes.get(key,0),int(row["final_size"]))
    allowed = {f"idx_{i}" for i in args.indices} if args.indices else set(sizes)
    jobs = []
    for item in json.loads((args.dataset/"tensor_manifest.json").read_text())["graphs"]:
        key = Path(item["source"]).stem
        if key in allowed:
            jobs.append((key,args.dataset/item["tensor"],sizes[key]))
    if len(jobs)!=len(allowed):
        raise ValueError("Incomplete graph inventory")
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"protocol.json").write_text(json.dumps(dict(solver="CaDiCaL 1.9.5", pysat=pysat_version,
        dataset=str(args.dataset), timed_run=str(args.run), graphs=len(jobs),
        query="Independent set larger than best delivered, using edge clauses and verified disjoint cliques.",
        timing="Post hoc exact calculation, outside every scored solve."),indent=2)+"\n")
    with ProcessPoolExecutor(max_workers=args.workers) as workers, (args.out/"optima.csv").open("w",newline="") as out:
        writer = csv.DictWriter(out,fieldnames=("graph_id","best_delivered","optimum","clique_upper","elapsed_s","larger_witness"))
        writer.writeheader()
        for i,row in enumerate(workers.map(verify,jobs),1):
            writer.writerow(row)
            if i%25==0 or i==len(jobs):
                out.flush()
                print(f"{i}/{len(jobs)} graphs",flush=True)


if __name__ == "__main__":
    main()
