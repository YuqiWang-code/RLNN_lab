"""P0.5: split the 0.2 s wall clock into named, measurable segments.

The 0.2 s cell (K=200, fp16+dense, PAVE reserve 40 ms) spends ~187 ms per
graph.  ``--v11_csv`` only records coarse buckets, and ``search_time_s``
starts before the incumbent decode, so the fixed part is opaque.  This
probe replays the same per-graph sequence with fine timers:

    load -> incumbent decode -> search loop -> PAVE

and re-runs the graph under two ablations that do not touch weights, K,
dtype, or the decode rhythm:

* ``full``        the deployed sequence (reference numbers);
* ``no_init``     skip the incumbent decode before the loop (does the
                  selected set change?  that decides whether removing it
                  is a numerics change);
* ``no_sync``     ``sync_each_block=False`` (does the per-block
                  ``cuda.synchronize`` really cost ~1.6 ms?).

``--fixed_steps`` is supported as well, so the same graph can be timed on
the step-matched path used by the t24 curve.

    python models/probe_p05_budget.py --test_path models/data/MIS_tensors/RB_small_author_test \
        --test_samples 20 --graphs 20 --budgets 0.2 --seeds 1017 --num_k 200 \
        --decode_every 1000 --decode_top 4 --dense --mixed_precision fp16 \
        --out outputs/v11/r2/p05/budget.json
"""

import json
import math
from pathlib import Path
from time import perf_counter

import torch

from eval_v2 import dataset_at, load_policy, sync
from eval_v7_budget import run_pave
from eval_v8_padded import build_parser, decode_top_best, shape_of
from mis_v1 import adjacency
from mis_v3_search import SearchGraph
from mis_v3_value import PAVEV3
from mis_v8_pad import PaddedSearch
from v7_runs import pave


def parse():
    parser = build_parser()
    parser.add_argument("--graphs", type=int, default=20,
                        help="graphs to time (starts at index 0)")
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/v11/r2/p05/budget.json"))
    parser.add_argument("--variants", default="full,no_init,no_sync",
                        help="comma separated subset of full,no_init,no_sync")
    parser.set_defaults(num_k=200, chunk=10, dense=True, mixed_precision="fp16",
                        reference_seed=0,
                        save_dir=Path("outputs/v2/none_seed0/checkpoints"),
                        decoders=["captured"], arms=["padded"])
    for action in parser._actions:
        if action.dest in ("result_dir", "reference_seed"):
            action.required = False
    return parser.parse_args()


def engine_for(solver, n_max, capacity, args):
    return PaddedSearch(solver, n_max=n_max, capacity=capacity, chunk=args.chunk,
                        device=solver.accelerator.device,
                        decode_chunk=args.decode_chunk, dense=args.dense)


def decode_once(solver, engine, adj, bias, n, args, pool):
    candidate = decode_top_best(solver.problem, engine.best_states, engine.best_energy,
                                adj, bias, n, args.decode_top)
    if candidate not in pool:
        pool.append(candidate)
    return candidate


def run_variant(solver, engine, ranker, graph, edges, graph_id, args, variant, budget):
    """Replays the eval_v8_padded per-graph block for one variant."""
    n = int(graph.num_nodes)
    bias = graph.b
    adj = adjacency(graph)
    seed = args.seeds[0]
    topology = SearchGraph(n, edges, seed=seed)
    reserve = min(args.post_reserve, args.reserve_frac * budget)
    pool = []
    record = {"graph_id": graph_id, "variant": variant, "budget": budget}

    sync(solver.accelerator.device)
    tick = perf_counter()
    deadline = tick + budget
    record["load_ms"] = 1e3 * engine.load(graph, edges, seed)
    init_size = None
    if variant != "no_init":
        started = perf_counter()
        init_size = decode_once(solver, engine, adj, bias, n, args, pool).bit_count()
        record["init_decode_ms"] = 1e3 * (perf_counter() - started)
    else:
        record["init_decode_ms"] = 0.0
    fixed_blocks = None
    cutoff = deadline - reserve
    if args.fixed_steps:
        cutoff = float("inf")
        fixed_blocks = math.ceil(args.fixed_steps / args.chunk)

    def decode(best):
        decode_once(solver, engine, adj, bias, n, args, pool)

    started = perf_counter()
    steps, decode_seconds, blocks = engine.run(
        cutoff, decode, skip_unchanged=False, decode_every=args.decode_every,
        max_blocks=fixed_blocks, sync_each_block=variant != "no_sync")
    sync(solver.accelerator.device)
    record["loop_ms"] = 1e3 * (perf_counter() - started)
    record["steps"] = steps
    record["blocks"] = blocks
    record["ms_per_step"] = record["loop_ms"] / max(steps, 1)
    record["decode_ms"] = 1e3 * decode_seconds

    pave_deadline = deadline if fixed_blocks is None else perf_counter() + reserve
    if pool:
        selected = max(pool, key=int.bit_count)
    else:
        selected = engine.decode_best(engine.best_states)
    started = perf_counter()
    final = run_pave(topology, selected, ranker, args, pave_deadline)
    record["pave_ms"] = 1e3 * (perf_counter() - started)
    record["search_size"] = selected.bit_count()
    record["final_size"] = final.bit_count()
    record["init_size"] = init_size
    record["total_ms"] = 1e3 * (perf_counter() - tick)
    if not is_independent_checked(selected, topology):
        record["illegal"] = graph_id
    return record


def is_independent_checked(mask, topology):
    from mis_v7_recombine import is_independent
    return is_independent(mask, topology.neighbors)


def summarise(records):
    keys = ("load_ms", "init_decode_ms", "loop_ms", "decode_ms", "pave_ms",
            "total_ms", "steps", "ms_per_step", "search_size", "final_size")
    out = {}
    for variant in sorted({record["variant"] for record in records}):
        chosen = [record for record in records if record["variant"] == variant]
        out[variant] = {key: sum(record[key] for record in chosen) / len(chosen)
                        for key in keys}
        out[variant]["graphs"] = len(chosen)
    base = out.get("full")
    if base:
        for variant, stats in out.items():
            if variant == "full":
                continue
            stats["delta_search_size"] = stats["search_size"] - base["search_size"]
            stats["delta_total_ms"] = stats["total_ms"] - base["total_ms"]
            stats["delta_steps"] = stats["steps"] - base["steps"]
    return out


def main():
    args = parse()
    variants = [part for part in args.variants.split(",") if part]
    solver = load_policy(args)
    solver.model.requires_grad_(False)
    checkpoint = torch.load(pave(args.reference_seed), map_location="cpu",
                            weights_only=False)
    ranker = PAVEV3(checkpoint["mode"], checkpoint.get("stats_dim", 24))
    ranker.load_state_dict(checkpoint["model_state"])
    ranker = ranker.cpu().eval()
    device = solver.accelerator.device
    dataset = dataset_at(args.test_path, solver.dtype, args.test_samples,
                         official=args.split == "test")
    n_max, capacity = shape_of(dataset, None)
    print(f"padded shape: nodes {n_max}, capacity {capacity}", flush=True)
    budget = args.budgets[0]
    records = []
    for index in range(min(args.graphs, len(dataset))):
        graph = dataset[index]
        edges = graph.edge_index
        graph_id = Path(dataset.files[index]).stem
        moved = graph.to(device)
        engine = engine_for(solver, n_max, capacity, args)
        engine.load(moved, edges, args.seeds[0])
        engine.run(perf_counter() + 0.03, lambda states: None)
        sync(device)
        for variant in variants:
            record = run_variant(solver, engine, ranker, moved, edges, graph_id,
                                 args, variant, budget)
            records.append(record)
            print("{:>12s} {:>8s} steps {:3d} load {:5.2f} init {:6.2f} "
                  "loop {:7.2f} pave {:6.2f} total {:7.2f} sel {:3d}".format(
                      graph_id, variant, record["steps"], record["load_ms"],
                      record["init_decode_ms"], record["loop_ms"],
                      record["pave_ms"], record["total_ms"],
                      record["search_size"]), flush=True)
        del engine
    summary = summarise(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": {"num_k": args.num_k, "chunk": args.chunk,
                          "decode_every": args.decode_every,
                          "decode_top": args.decode_top,
                          "dense": bool(args.dense),
                          "mixed_precision": args.mixed_precision,
                          "budget": budget, "fixed_steps": args.fixed_steps,
                          "reserve_frac": args.reserve_frac,
                          "post_reserve": args.post_reserve,
                          "graphs": args.graphs, "seed": args.seeds[0]},
               "summary": summary, "records": records}
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
