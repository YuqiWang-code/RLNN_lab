"""Does exact recombination add anything on top of a population of RLSA solutions?

RLSA at its paper operating point (``num_t 300 num_k 200 num_d 5``) reaches
SIZE 19.962 at 0.201 s per graph.  It reports the best of its ``num_k`` decoded
particles and throws the other 199 away.  Our own particles are local optima of a
much weaker searcher (best 19.35 at the census probe), so they say little about
what recombination can do once the population itself is good.

This probe asks, on the same graphs and seeds:

1. how much exact Koenig recombination gains over the best single particle when
   the population is a set of RLSA local optima sitting at ~19.9;
2. what it costs in wall time, so it can be placed inside the one-second budget
   that also pays for the search.

Two deterministic policies:

``chain``   sort the population by size, fold the incumbent into the next
            solution one at a time (``K`` matchings);
``oracle``  exhaustive over pairs of the top ``--oracle_pool`` solutions, i.e.
            the ceiling any pairwise policy could reach.
"""
import csv
import json
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from eval_rlsa_curve import curve_parser
from eval_v2 import TEST, TEST_CACHE, dataset_at, sync
from mis_v3_search import SearchGraph
from mis_v7_recombine import adjacency_masks, is_independent, recombine
from solvers import RLSA


@torch.no_grad()
def rlsa_population(solver, graph, generator):
    """The RLSA loop of ``RLSA.evaluate_single``, keeping the whole population."""
    x = torch.randint(0, 2, (graph.num_nodes, solver.num_k),
                      device=solver.accelerator.device,
                      dtype=graph.edge_weight.dtype, generator=generator)
    A = torch.sparse_coo_tensor(
        graph.edge_index, graph.edge_weight,
        torch.Size((graph.num_nodes, graph.num_nodes)),
    ).to_sparse_csr()
    energy, grad = solver.problem.energy_func(A, graph.b, x, True)
    best_sol, best_energy = x.clone(), energy.clone()
    for epoch in range(solver.num_t):
        tau = solver.tau0 * (1 - epoch / solver.num_t)
        delta = grad * (2 * x - 1) / 2
        term2 = -torch.kthvalue(-delta, solver.num_d, dim=0, keepdim=True).values
        flip_prob = torch.sigmoid((delta - term2) / tau)
        rr = torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        x = torch.where(rr < flip_prob, 1 - x, x)
        energy, grad = solver.problem.energy_func(
            A, graph.b, x, True if epoch < solver.num_t - 1 else False)
        improves = energy < best_energy
        best_sol = torch.where(improves.unsqueeze(0).repeat(graph.num_nodes, 1), x, best_sol)
        best_energy = torch.where(improves, energy, best_energy)
    decoded = solver.problem.decode_gradient_flip(best_sol, A, graph.b, return_solutions=True)
    packed = np.packbits(decoded.t().to(torch.uint8).cpu().numpy(), axis=1, bitorder="little")
    return [int.from_bytes(row.tobytes(), "little") for row in packed], int(x.shape[1])


def chain(masks, neighbours, passes):
    """Fold the incumbent into the population, largest solutions first."""
    order = sorted(masks, key=int.bit_count, reverse=True)
    incumbent = order[0]
    calls = 0
    for _ in range(passes):
        changed = False
        for mask in order[1:]:
            candidate = recombine(incumbent, mask, neighbours).candidate
            calls += 1
            if candidate.bit_count() > incumbent.bit_count():
                incumbent = candidate
                changed = True
        if not changed:
            break
    return incumbent, calls


def oracle(masks, neighbours, pool):
    top = sorted(masks, key=int.bit_count, reverse=True)[:pool]
    best = max(masks, key=int.bit_count)
    calls = 0
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            candidate = recombine(top[i], top[j], neighbours).candidate
            calls += 1
            if candidate.bit_count() > best.bit_count():
                best = candidate
    return best, calls


def main():
    parser = curve_parser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--oracle_pool", type=int, default=40)
    parser.add_argument("--chain_passes", type=int, default=2)
    args = parser.parse_args()
    if args.test_path is None:
        args.test_path = str(TEST_CACHE if (TEST_CACHE / "tensor_manifest.json").is_file() else TEST)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    solver = RLSA(args)
    solver.prepare()
    device = solver.accelerator.device
    dataset = dataset_at(args.test_path, solver.dtype, args.samples, official=args.split == "test")
    seed = args.seeds[0]

    rows = []
    for index in range(len(dataset)):
        edges = dataset[index].edge_index
        graph = dataset[index].to(device)
        neighbours = SearchGraph(int(graph.num_nodes), edges, seed=seed).neighbors
        generator = torch.Generator(device=device).manual_seed(seed)
        sync(device)
        started = perf_counter()
        masks, width = rlsa_population(solver, graph, generator)
        sync(device)
        search_seconds = perf_counter() - started
        if not all(is_independent(mask, neighbours) for mask in masks):
            raise SystemExit(f"idx_{index}: RLSA produced an illegal set")

        started = perf_counter()
        chained, chain_calls = chain(masks, neighbours, args.chain_passes)
        chain_ms = 1e3 * (perf_counter() - started)
        started = perf_counter()
        pooled, oracle_calls = oracle(masks, neighbours, args.oracle_pool)
        oracle_ms = 1e3 * (perf_counter() - started)

        best = max(mask.bit_count() for mask in masks)
        union = 0
        for mask in masks:
            union |= mask
        rows.append(dict(
            graph_id=f"idx_{index}", num_t=args.num_t, num_k=width,
            best_single=best, chain_size=chained.bit_count(), oracle_size=pooled.bit_count(),
            chain_gain=chained.bit_count() - best, oracle_gain=pooled.bit_count() - best,
            union_size=union.bit_count(), search_s=round(search_seconds, 4),
            chain_ms=round(chain_ms, 2), oracle_ms=round(oracle_ms, 2),
            chain_calls=chain_calls, oracle_calls=oracle_calls,
        ))
        print(f"idx_{index} T{args.num_t} best {best} chain {rows[-1]['chain_size']} "
              f"(+{rows[-1]['chain_gain']}) oracle {rows[-1]['oracle_size']} "
              f"(+{rows[-1]['oracle_gain']}) search {search_seconds:.3f}s "
              f"chain {rows[-1]['chain_ms']}ms oracle {rows[-1]['oracle_ms']}ms", flush=True)

    path = args.result_dir / f"rlsa_recombine_n{args.num_k}.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    keys = ("best_single", "chain_size", "oracle_size", "chain_gain", "oracle_gain",
            "union_size", "search_s", "chain_ms", "oracle_ms")
    summary = {key: mean(row[key] for row in rows) for key in keys}
    summary.update(graphs=len(rows), num_t=args.num_t, num_k=args.num_k, num_d=args.num_d,
                   oracle_pool=args.oracle_pool)
    (args.result_dir / f"rlsa_recombine_n{args.num_k}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
