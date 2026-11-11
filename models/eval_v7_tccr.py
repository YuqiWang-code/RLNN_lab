"""Fixed-work pipeline evaluation: does trajectory recombination survive frozen PAVE?

The census showed that the K=20 A.3 solutions of the frozen search can be
recombined into a strictly larger legal set on most graphs.  The deployment
question is different: the frozen baseline is ``B + PAVE``, so recombination only
matters if it adds something *on top of* the frozen post-processing, and if the
cost fits the protocol.

For every graph and search seed this script records four legal sets:

    a3          A.3 argmax over the K=20 particles      (the frozen "none" arm)
    recombine   exhaustive exact recombination of all K*(K-1)/2 pairs
    pave        frozen PAVE-Stats H1 started from a3
    recombine+pave  the same frozen PAVE started from the recombined set

Nothing is trained.  PAVE, its parameters and its step budget are untouched; the
only new work is the recombination pass.

    python -u models/eval_v7_tccr.py --reference_seed 0 --save_dir outputs/v2/none_seed0/checkpoints \
        --result_dir outputs/v7_tccr/pipeline --split validation --test_samples 100 --seeds 1017 1019 1021
"""
import json
from math import inf
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

import mis_v7_recombine as recombine
from diagnose_v7_census import build_parser, captured_best_states, solution_masks, validate, write_csv
from eval_v2 import dataset_at, load_policy, sync, write_run
from mis_v1 import adjacency
from mis_v3_search import SearchGraph, platform_search
from mis_v3_value import PAVEV3
from v7_runs import digest, pave, provenance


def main():
    import argparse

    parser = build_parser()
    parser.set_defaults(pair_budget=0)  # 0 evaluates every pair; a positive value keeps the Hamming prefix
    args = parser.parse_args()
    validate(parser, args)
    if args.phase != "recombine":
        parser.error("the pipeline evaluation reuses the census parser with --phase recombine")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    output = args.result_dir / "pipeline"
    output.mkdir(exist_ok=True)

    solver = load_policy(args)
    solver.cuda_graph_search = args.cuda_graph
    solver.model.requires_grad_(False)
    checkpoint = torch.load(pave(args.reference_seed), map_location="cpu", weights_only=False)
    ranker = PAVEV3(checkpoint["mode"], checkpoint.get("stats_dim", 24))
    ranker.load_state_dict(checkpoint["model_state"])
    ranker = ranker.cpu().eval()
    device = solver.accelerator.device
    dataset = dataset_at(args.test_path, solver.dtype, args.test_samples, official=args.split == "test")

    warm = dataset[0].to(device)
    warm_adj = adjacency(warm)
    captured_best_states(solver, warm, warm_adj, args.seeds[0])
    sync(device)

    def run_pave(topology, selected):
        result = platform_search(topology, selected, choose=ranker.choose,
                                 outer_rounds=args.outer_rounds,
                                 initial_rounds=args.exchange_rounds, horizon=args.horizon,
                                 candidate_cap=args.candidate_cap,
                                 improving_cap=args.improving_cap,
                                 repair_rounds=args.repair_rounds, anchored=False, deadline=inf)
        return result.best

    rows = []
    started = perf_counter()
    for index, graph in enumerate(dataset):
        graph_id = f"idx_{index}" if args.split == "test" else Path(dataset.files[index]).stem
        edges = graph.edge_index
        graph = graph.to(device)
        adj = adjacency(graph)
        for seed in args.seeds:
            topology = SearchGraph(graph.num_nodes, edges, seed=seed)
            neighbours = topology.neighbors
            search_started = perf_counter()
            best_states, _ = captured_best_states(solver, graph, adj, seed)
            sync(device)
            decode_started = perf_counter()
            masks = solution_masks(solver.problem, best_states, adj, graph.b)
            decode_seconds = perf_counter() - decode_started
            a3 = max(masks, key=int.bit_count)
            search_seconds = perf_counter() - search_started
            recombine_started = perf_counter()
            if args.pair_budget:
                order = recombine.select_pairs(masks, "hamming")[:args.pair_budget]
                merged = a3
                for i, j in order:
                    candidate = recombine.recombine(masks[i], masks[j], neighbours).candidate
                    if candidate.bit_count() > merged.bit_count():
                        merged = candidate
                merged_gain = merged.bit_count() - a3.bit_count()
            else:
                merged, merged_gain, _, _ = recombine.best_recombination(masks, neighbours)
            recombine_seconds = perf_counter() - recombine_started
            for mask in (a3, merged):
                if not recombine.is_independent(mask, neighbours):
                    raise ValueError(f"{graph_id}: illegal set before PAVE")
            pave_started = perf_counter()
            paved_a3 = run_pave(topology, a3)
            pave_a3_seconds = perf_counter() - pave_started
            pave_started = perf_counter()
            paved_merged = run_pave(topology, merged)
            pave_merged_seconds = perf_counter() - pave_started
            for mask in (paved_a3, paved_merged):
                if not recombine.is_independent(mask, neighbours):
                    raise ValueError(f"{graph_id}: PAVE returned an illegal set")
            rows.append(dict(graph_id=graph_id, search_seed=seed, num_k=len(masks),
                             a3_size=a3.bit_count(), recombine_size=merged.bit_count(),
                             recombine_gain=merged_gain,
                             pave_size=paved_a3.bit_count(),
                             recombine_pave_size=paved_merged.bit_count(),
                             recombine_pave_gain=paved_merged.bit_count() - paved_a3.bit_count(),
                             pave_gain=paved_a3.bit_count() - a3.bit_count(),
                             search_ms=round(1e3 * search_seconds, 2),
                             decode_ms=round(1e3 * decode_seconds, 2),
                             recombine_ms=round(1e3 * recombine_seconds, 2),
                             pave_ms=round(1e3 * pave_a3_seconds, 2),
                             recombine_pave_ms=round(1e3 * pave_merged_seconds, 2),
                             pairs=(len(masks) * (len(masks) - 1) // 2) if not args.pair_budget else args.pair_budget))
            sync(device)
        if (index + 1) % 20 == 0 or index + 1 == len(dataset):
            print(f"{index + 1}/{len(dataset)} graphs, "
                  f"a3={mean(row['a3_size'] for row in rows):.4f}, "
                  f"recombine={mean(row['recombine_size'] for row in rows):.4f}, "
                  f"pave={mean(row['pave_size'] for row in rows):.4f}, "
                  f"recombine+pave={mean(row['recombine_pave_size'] for row in rows):.4f}", flush=True)
    finished = perf_counter()
    write_csv(output / f"pipeline_ref{args.reference_seed}.csv", rows)
    metadata = dict(version="v7_tccr", status="completed", mode="pipeline",
                           ref=args.reference_seed, seeds=args.seeds, split=args.split,
                           samples=len(dataset), graphs=len(dataset),
                           wall_time_s=round(finished - started, 2),
                           pair_budget=args.pair_budget,
                           checkpoint_sha256=digest(Path(args.save_dir) / "best.pt"),
                           mean_a3=mean(row["a3_size"] for row in rows),
                           mean_recombine=mean(row["recombine_size"] for row in rows),
                           mean_pave=mean(row["pave_size"] for row in rows),
                           mean_recombine_pave=mean(row["recombine_pave_size"] for row in rows),
                    config=vars(args), **provenance(args.reference_seed))
    (output / f"run_ref{args.reference_seed}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"wrote {output} in {finished - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
