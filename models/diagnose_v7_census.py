"""TCCR census: how much improvement do the two v7 operators actually reach?

Phase 1 keeps the K=20 legal sets that the frozen RLNN search already produces and
asks whether exact recombination of two of them beats the best of them.  Phase 2
takes the A.3 solution and the frozen-PAVE solution and asks whether a
conflict-certified release region (``U`` of size 2/3/4, ``R(U)``, exact local MIS)
can grow them by at least one node.

Nothing is trained and nothing is written back into the search.  The protocol is
the frozen one: K20, T100, FP32/CUDA-graph optional, A.3 decoding, PAVE-Stats H1.
Every legal set is checked twice, once with ``SearchGraph.feasible`` and once with
an independent bitset test, so the census cannot silently report on illegal input.

    python -u models/diagnose_v7_census.py --reference_seed 0 --result_dir outputs/v7_tccr \
        --split validation --samples 100 --seeds 1017 1019 1021 --phase both
"""
import csv
import json
from math import inf
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

import mis_v7_conflict as conflict
import mis_v7_recombine as recombine
from eval_v2 import dataset_at, load_policy, sync, write_run
from mis_v1 import adjacency
from mis_v3_search import SearchGraph, platform_search
from mis_v3_value import PAVEV3
from v7_runs import digest, pave, provenance

STRATEGIES = ("random", "hamming", "top_size", "matching_cost")


def build_parser():
    import eval_v3

    parser = eval_v3.build_parser()
    for action in parser._actions:
        if action.dest == "label":
            action.required = False
    parser.set_defaults(split="validation", post=["none"], mixed_precision="no",
                        label="v7_tccr_census")
    parser.add_argument("--reference_seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1017, 1019, 1021],
                        help="Search seeds; the frozen baseline used 1017")
    parser.add_argument("--phase", choices=("recombine", "conflict", "both"), default="both")
    parser.add_argument("--region_sizes", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--pair_budget", type=int, default=8,
                        help="Calls spent by each non-oracle pair strategy")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--check_search", action="store_true",
                        help="Also run eval_v2.search and assert the captured solution matches it")
    parser.add_argument("--check_graphs", type=int, default=3)
    return parser


def validate(parser, args):
    expected = dict(num_k=20, num_d=5, beta=1.02, horizon=2, candidate_cap=32,
                    exchange_rounds=3, outer_rounds=8, improving_cap=48, repair_rounds=4)
    for key, value in expected.items():
        if getattr(args, key) != value:
            parser.error(f"the census fixes {key}={value}")
    if args.num_t != 100:
        parser.error("the census runs at T100")
    if args.time_budget_s is not None or args.extended_search:
        parser.error("the census is a fixed-work diagnosis, not a deadline run")


@torch.no_grad()
def captured_best_states(solver, graph, adj, seed):
    """Replay the T100 search and keep each particle's best-energy state.

    The loop mirrors ``eval_v2.search`` line for line, including the call to
    ``solver.policy_topology`` that produces the model-side normalised adjacency,
    so the captured states are the ones the frozen evaluation decodes.
    ``--check_search`` asserts that the best decoded size agrees with
    ``eval_v2.search``.
    """
    device = graph.b.device
    generator = torch.Generator(device=device).manual_seed(seed)
    problem = solver.problem
    topology = solver.policy_topology(graph, adj)
    if getattr(solver, "cuda_graph_search", False):
        from capture_mis import captured_states
        states = captured_states(solver, graph, adj, topology, generator)
    elif getattr(solver, "v6_frequency", False):
        states = solver.sample_states(graph, adj, generator=generator, normalized=topology,
                                      with_energy=True)
    else:
        states = ((x, problem.energy_func(adj, graph.b, x)[0])
                  for x in solver.sample_states(graph, adj, generator=generator, normalized=topology))
    best_states, best_energy = next(states)
    steps = 0
    for step in range(1, solver.num_t + 1):
        state, energy = next(states)
        improves = energy < best_energy
        best_states = torch.where(improves.unsqueeze(0), state, best_states)
        best_energy = torch.where(improves, energy, best_energy)
        steps = step
    return best_states, steps


def solution_masks(problem, states, adj, bias):
    """A.3 on every particle, as integer bitsets.

    ``best_decoded`` keeps only the argmax column; the census needs all of them.
    """
    decoded = problem.decode_gradient_flip(states, adj, bias, return_solutions=True)
    packed = np.packbits(decoded.t().to(torch.uint8).cpu().numpy(), axis=1, bitorder="little")
    return [int.from_bytes(row.tobytes(), "little") for row in packed]


def apply_pairs(masks, neighbours, order, budget):
    """Greedy chaining of at most ``budget`` recombinations in the given order."""
    incumbent = max(masks, key=int.bit_count)
    used = 0
    for i, j in order[:budget]:
        candidate = recombine.recombine(masks[i], masks[j], neighbours).candidate
        used += 1
        if candidate.bit_count() > incumbent.bit_count():
            incumbent = candidate
    return incumbent, used


def recombine_row(graph_id, seed, masks, neighbours, pair_budget, decode_seconds):
    sizes = [mask.bit_count() for mask in masks]
    union = 0
    for mask in masks:
        union |= mask
    hamming, jaccard = [], []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            overlap = (masks[i] | masks[j]).bit_count()
            hamming.append((masks[i] ^ masks[j]).bit_count())
            jaccard.append((masks[i] & masks[j]).bit_count() / overlap if overlap else 1.0)
    started = perf_counter()
    best_candidate, best_gain, best_pair, pair_gains = recombine.best_recombination(masks, neighbours)
    oracle_seconds = perf_counter() - started
    assert recombine.is_independent(best_candidate, neighbours)
    baseline = max(sizes)
    assert best_candidate.bit_count() - baseline == best_gain
    arms = {}
    for strategy in STRATEGIES:
        incumbent, _ = apply_pairs(masks, neighbours, recombine.select_pairs(masks, strategy), pair_budget)
        arms[strategy] = incumbent.bit_count() - max(sizes)
    pair_count = len(pair_gains)
    return dict(graph_id=graph_id, search_seed=seed, num_k=len(masks),
                distinct=len(set(masks)), best_size=max(sizes), mean_size=round(mean(sizes), 4),
                min_size=min(sizes), max_size=max(sizes), union_size=union.bit_count(),
                hamming_mean=round(mean(hamming), 4), jaccard_mean=round(mean(jaccard), 4),
                oracle_gain=best_gain, oracle_pair=str(best_pair),
                pairwise_gain_max=max(pair_gains or [0]),
                improving_pairs=sum(gain > 0 for gain in pair_gains),
                useful_pairs=count_useful_pairs(masks, neighbours, baseline),
                pair_count=pair_count, pair_gain_sum=sum(pair_gains),
                oracle_us_per_pair=round(1e6 * oracle_seconds / max(pair_count, 1), 2),
                decode_ms=round(1e3 * decode_seconds, 2),
                **{f"arm_{name}": value for name, value in arms.items()})


def count_useful_pairs(masks, neighbours, baseline):
    """Pairs whose recombined set beats the best single solution of the list."""
    useful = 0
    for i, j in recombine.pairs(len(masks)):
        if recombine.recombine(masks[i], masks[j], neighbours).size > baseline:
            useful += 1
    return useful


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate(parser, args)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    census_dir = args.result_dir / "census"
    census_dir.mkdir(exist_ok=True)

    solver = load_policy(args)
    solver.cuda_graph_search = args.cuda_graph
    solver.model.requires_grad_(False)
    ranker = None
    if args.phase in ("conflict", "both"):
        checkpoint = torch.load(pave(args.reference_seed), map_location="cpu", weights_only=False)
        ranker = PAVEV3(checkpoint["mode"], checkpoint.get("stats_dim", 24))
        ranker.load_state_dict(checkpoint["model_state"])
        ranker = ranker.cpu().eval()
    device = solver.accelerator.device
    dataset = dataset_at(args.test_path, solver.dtype, args.test_samples, official=args.split == "test")
    num_t = solver.num_t

    warm = dataset[0].to(device)
    warm_adj = adjacency(warm)
    captured_best_states(solver, warm, warm_adj, args.seeds[0])
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    recombine_rows, conflict_rows, checks = [], [], []
    started = perf_counter()
    for index, graph in enumerate(dataset):
        graph_id = f"idx_{index}" if args.split == "test" else Path(dataset.files[index]).stem
        edges = graph.edge_index
        graph = graph.to(device)
        adj = adjacency(graph)
        for seed in args.seeds:
            topology = SearchGraph(graph.num_nodes, edges, seed=seed)
            neighbours = topology.neighbors
            tick = perf_counter()
            best_states, steps = captured_best_states(solver, graph, adj, seed)
            sync(device)
            decode_started = perf_counter()
            masks = solution_masks(solver.problem, best_states, adj, graph.b)
            decode_seconds = perf_counter() - decode_started
            for mask in masks:
                if not topology.feasible(mask):
                    raise ValueError(f"{graph_id}: A.3 returned a conflicting set")
                if not recombine.is_independent(mask, neighbours):
                    raise ValueError(f"{graph_id}: bitset check disagrees with SearchGraph")
            a3_best = max(masks, key=int.bit_count)
            if args.check_search and index < args.check_graphs and seed == args.seeds[0]:
                from eval_v2 import search as v2_search
                reference, reference_steps = v2_search(solver, graph, topology, seed)
                checks.append(dict(graph_id=graph_id, seed=seed,
                                   captured_size=a3_best.bit_count(),
                                   search_size=reference.bit_count(),
                                   captured_steps=steps, search_steps=reference_steps,
                                   match=int(a3_best == reference)))
            if args.phase in ("recombine", "both"):
                recombine_rows.append(recombine_row(graph_id, seed, masks, neighbours,
                                                    args.pair_budget, decode_seconds))
            if args.phase in ("conflict", "both"):
                census_cache = {}
                pave_started = perf_counter()
                result = platform_search(topology, a3_best, choose=ranker.choose,
                                         outer_rounds=args.outer_rounds,
                                         initial_rounds=args.exchange_rounds, horizon=args.horizon,
                                         candidate_cap=args.candidate_cap,
                                         improving_cap=args.improving_cap,
                                         repair_rounds=args.repair_rounds, anchored=False,
                                         deadline=inf)
                pave_final = result.best
                pave_seconds = perf_counter() - pave_started
                if not recombine.is_independent(pave_final, neighbours):
                    raise ValueError(f"{graph_id}: PAVE returned a conflicting set")
                for layer, initial, final in (("a3", a3_best, a3_best), ("pave", a3_best, pave_final)):
                    cached = census_cache.get(final)
                    if cached is None:
                        census_started = perf_counter()
                        stats = conflict.census(final, neighbours, graph.num_nodes,
                                                sizes=tuple(args.region_sizes))
                        census_seconds = perf_counter() - census_started
                        census_cache[final] = (stats, census_seconds)
                    else:
                        stats, census_seconds = cached
                    row = dict(graph_id=graph_id, search_seed=seed, layer=layer,
                               base_size=initial.bit_count(), final_size=final.bit_count(),
                               pave_gain=final.bit_count() - initial.bit_count(),
                               stalled=int(final.bit_count() == initial.bit_count()),
                               a3_steps=steps, search_ms=round(1e3 * (perf_counter() - tick), 2),
                               pave_ms=round(1e3 * pave_seconds, 2),
                               census_ms=round(1e3 * census_seconds, 2))
                    for size in args.region_sizes:
                        entry = stats[size]
                        best = entry["best"]
                        row[f"regions_{size}"] = entry["regions"]
                        row[f"improving_{size}"] = entry["improving"]
                        row[f"gain_{size}"] = entry["best_gain"]
                        row[f"removed_{size}"] = best.removed if best else 0
                        row[f"release_{size}"] = best.release.bit_count() if best else 0
                        row[f"window_{size}"] = best.window.bit_count() if best else 0
                        row[f"exact_{size}"] = int(entry["exact"])
                    conflict_rows.append(row)
            sync(device)
        print(f"{index + 1}/{len(dataset)} {graph_id} done", flush=True)

    finished = perf_counter()
    metadata = dict(version="v7_tccr", status="completed", mode="census", phase=args.phase,
                    ref=args.reference_seed, seeds=args.seeds, split=args.split,
                    samples=len(dataset), num_t=num_t, num_k=args.num_k,
                    cuda_graph=args.cuda_graph, pair_budget=args.pair_budget,
                    region_sizes=args.region_sizes, wall_time_s=round(finished - started, 2),
                    checkpoint_sha256=digest(Path(args.save_dir) / "best.pt"),
                    config=vars(args), checks=checks,
                    **provenance(args.reference_seed))
    if recombine_rows:
        write_csv(census_dir / f"recombine_ref{args.reference_seed}.csv", recombine_rows)
    if conflict_rows:
        write_csv(census_dir / f"conflict_ref{args.reference_seed}.csv", conflict_rows)
    if checks:
        (args.result_dir / "checks").mkdir(parents=True, exist_ok=True)
        write_run(args.result_dir / "checks", dict(version="v7_tccr", status="completed",
                                                   mode="capture_check",
                                                   match=int(all(row["match"] for row in checks)),
                                                   rows=checks))
    (census_dir / f"run_ref{args.reference_seed}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"wrote {census_dir} in {finished - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
