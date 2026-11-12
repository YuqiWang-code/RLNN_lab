"""Deadline (budget) evaluation: does recombination pay for itself inside 0.5 s / 1.0 s?

The fixed-work census showed that recombining the K=20 A.3 solutions adds about
+0.09 on top of frozen PAVE when the search is already finished.  In a deadline
protocol that time has to come out of the search, so this script runs two arms
with the *same* total budget on the same graph and seed:

    baseline    search until deadline - 0.05 s, then frozen PAVE until deadline
    recombine   search until deadline - 0.07 s, exact recombination (<= 0.02 s),
                then the same frozen PAVE until deadline

Both arms decode every 10 search steps exactly like the frozen protocol; the
decode is a full K-particle A.3 call either way and the argmax is free, so the
recombination arm does not pay for an extra decode.  The recombination itself is
screened by ``mis_v7_fast``: one batched matching gives the exact gain of all 190
pairs and only the winner goes through the exact operator.

    python -u models/eval_v7_budget.py --reference_seed 0 --save_dir outputs/v2/none_seed0/checkpoints \
        --result_dir outputs/v7_tccr/budget --split validation --test_samples 100 \
        --seeds 1017 1019 1021 --budgets 0.5 1.0
"""
import csv
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

import mis_v7_fast as fast
import mis_v7_recombine as recombine
from diagnose_v7_census import build_parser, solution_masks
from eval_v2 import dataset_at, load_policy, sync, write_run
from mis_v1 import adjacency
from mis_v3_search import SearchGraph, platform_search
from mis_v3_value import PAVEV3
from v7_runs import digest, pave, provenance

DECODE_INTERVAL = 10
POST_RESERVE = 0.05
RECOMBINE_RESERVE = 0.02


def build_budget_parser():
    parser = build_parser()
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--recombine_reserve", type=float, default=RECOMBINE_RESERVE)
    parser.add_argument("--post_reserve", type=float, default=POST_RESERVE)
    return parser


@torch.no_grad()
def deadline_search(solver, graph, adj, seed, cutoff, normalized=None):
    """Frozen T-interval search, but stop at ``cutoff`` and keep every decode.

    Mirrors ``eval_v2.search``: the incumbent improves only on the 10-step decodes,
    the chunk time of the last decode is charged before the next step, and the
    per-particle best states are the ones recombination consumes.
    """
    entered = perf_counter()
    device = graph.b.device
    generator = torch.Generator(device=device).manual_seed(seed)
    problem = solver.problem
    if normalized is None:
        normalized = solver.policy_topology(graph, adj)
    states = ((x, problem.energy_func(adj, graph.b, x)[0])
              for x in solver.sample_states(graph, adj, generator=generator, normalized=normalized))
    best_states, best_energy = next(states)
    masks = None
    incumbent = 0
    steps = 0
    chunk_time = 0.0
    chunk_start = perf_counter()
    decode_seconds = 0.0

    def decode():
        nonlocal decode_seconds
        started = perf_counter()
        decoded = solution_masks(problem, best_states, adj, graph.b)
        decode_seconds += perf_counter() - started
        return decoded

    masks = decode()
    incumbent = max(masks, key=int.bit_count)
    prelude = perf_counter() - entered
    for step in range(1, solver.num_t + 1):
        if perf_counter() + chunk_time >= cutoff:
            break
        state, energy = next(states)
        improves = energy < best_energy
        best_states = torch.where(improves.unsqueeze(0), state, best_states)
        best_energy = torch.where(improves, energy, best_energy)
        steps = step
        if step % DECODE_INTERVAL == 0:
            masks = decode()
            now = perf_counter()
            candidate = max(masks, key=int.bit_count)
            if candidate.bit_count() > incumbent.bit_count():
                incumbent = candidate
            chunk_time = now - chunk_start
            chunk_start = now
    return incumbent, masks, steps, decode_seconds, prelude


def run_pave(topology, selected, ranker, args, deadline):
    result = platform_search(topology, selected, choose=ranker.choose,
                             outer_rounds=args.outer_rounds, initial_rounds=args.exchange_rounds,
                             horizon=args.horizon, candidate_cap=args.candidate_cap,
                             improving_cap=args.improving_cap, repair_rounds=args.repair_rounds,
                             anchored=False, deadline=deadline)
    return result.best


def main():
    parser = build_budget_parser()
    args = parser.parse_args()
    if args.num_t != 100:
        parser.error("the budget run keeps the frozen T100 configuration for the non-deadline path")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    output = args.result_dir / "budget"
    output.mkdir(exist_ok=True)

    solver = load_policy(args)
    solver.model.requires_grad_(False)
    checkpoint = torch.load(pave(args.reference_seed), map_location="cpu", weights_only=False)
    ranker = PAVEV3(checkpoint["mode"], checkpoint.get("stats_dim", 24))
    ranker.load_state_dict(checkpoint["model_state"])
    ranker = ranker.cpu().eval()
    device = solver.accelerator.device
    dataset = dataset_at(args.test_path, solver.dtype, args.test_samples, official=args.split == "test")
    extended = 10000
    # Warm up kernels, the A.3 decode and PAVE on the first graph: without this the
    # first measured arm spends its whole deadline compiling.
    solver.num_t = extended
    warm = dataset[0].to(device)
    warm_adj = adjacency(warm)
    warm_topology = SearchGraph(warm.num_nodes, dataset[0].edge_index, seed=args.seeds[0])
    warm_selected, warm_masks, _, _, _ = deadline_search(solver, warm, warm_adj, args.seeds[0],
                                                         perf_counter() + 0.2)
    run_pave(warm_topology, warm_selected, ranker, args, perf_counter() + 0.1)
    fast.oracle(warm_masks, warm_topology.neighbors)
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rows = []
    started = perf_counter()
    for index, graph in enumerate(dataset):
        graph_id = f"idx_{index}" if args.split == "test" else Path(dataset.files[index]).stem
        edges = graph.edge_index
        graph = graph.to(device)
        adj = adjacency(graph)
        for seed in args.seeds:
            for budget in args.budgets:
                topology = SearchGraph(graph.num_nodes, edges, seed=seed)
                neighbours = topology.neighbors
                solver.num_t = extended
                normalized = solver.policy_topology(graph, adj)
                # A throwaway pass of the same shape so both arms see the same
                # allocator and kernel cache.  A short one is not enough: the first
                # pass over a new graph runs at tens of ms/step instead of ~2 ms.
                deadline_search(solver, graph, adj, seed, perf_counter() + 0.25, normalized)
                sync(device)
                # Alternate which arm runs first so residual warm-up and GPU
                # contention do not systematically favour either side.
                base_order = ("baseline", "recombine") if (index + seed) % 2 == 0 else ("recombine", "baseline")
                for repeat in (0, 1):
                    arms = base_order if repeat == 0 else base_order[::-1]
                    for arm in arms:
                        reserve = args.post_reserve + (args.recombine_reserve if arm == "recombine" else 0.0)
                        sync(device)
                        tick = perf_counter()
                        deadline = tick + budget
                        cutoff = deadline - reserve
                        selected, masks, steps, decode_seconds, prelude = deadline_search(
                            solver, graph, adj, seed, cutoff, normalized)
                        sync(device)
                        search_seconds = perf_counter() - tick
                        recombine_seconds = 0.0
                        base_size = selected.bit_count()
                        if arm == "recombine":
                            recombine_started = perf_counter()
                            matrix = fast.adjacency_matrix(neighbours)
                            recombine_deadline = min(recombine_started + args.recombine_reserve, deadline)
                            merged, gain, pair, _ = fast.oracle_with_deadline(
                                masks, neighbours, matrix, deadline=recombine_deadline)
                            sync(device)
                            recombine_seconds = perf_counter() - recombine_started
                            if merged.bit_count() > selected.bit_count():
                                selected = merged
                        final = run_pave(topology, selected, ranker, args, deadline)
                        sync(device)
                        seconds = perf_counter() - tick
                        if not recombine.is_independent(final, neighbours):
                            raise ValueError(f"{graph_id}: illegal set in arm {arm}")
                        rows.append(dict(graph_id=graph_id, search_seed=seed, budget=budget, arm=arm,
                                         repeat=repeat, first=int(arms[0] == arm),
                                         final_size=final.bit_count(),
                                         search_size=base_size,
                                         recombine_gain=selected.bit_count() - base_size,
                                         pave_gain=final.bit_count() - selected.bit_count(),
                                         steps=steps,
                                         solve_time_s=round(seconds, 4),
                                         search_time_s=round(search_seconds, 4),
                                         recombine_time_s=round(recombine_seconds, 4),
                                         decode_time_s=round(decode_seconds, 4),
                                         prelude_s=round(prelude, 4),
                                         ms_per_step=round(1e3 * search_seconds / max(steps, 1), 3),
                                         deadline_overrun=int(seconds > budget)))
            solver.num_t = 100
        if (index + 1) % 20 == 0 or index + 1 == len(dataset):
            summary = " ".join(
                f"{budget}s:{mean(row['final_size'] for row in rows if row['budget'] == budget and row['arm'] == 'baseline'):.3f}"
                f"/{mean(row['final_size'] for row in rows if row['budget'] == budget and row['arm'] == 'recombine'):.3f}"
                for budget in args.budgets)
            print(f"{index + 1}/{len(dataset)} graphs base/recombine {summary}", flush=True)
    metadata = dict(version="v7_tccr", status="completed", mode="budget",
                           ref=args.reference_seed, seeds=args.seeds, budgets=args.budgets,
                           samples=len(dataset), post_reserve=args.post_reserve,
                           recombine_reserve=args.recombine_reserve,
                           wall_time_s=round(perf_counter() - started, 2),
                    config=vars(args), **provenance(args.reference_seed))
    (output / f"run_ref{args.reference_seed}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    path = output / f"budget_ref{args.reference_seed}.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for budget in args.budgets:
        for arm in ("baseline", "recombine"):
            sizes = [row["final_size"] for row in rows if row["budget"] == budget and row["arm"] == arm]
            print(f"budget={budget} {arm}: mean {mean(sizes):.4f} over {len(sizes)} runs", flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
