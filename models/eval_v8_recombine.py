"""Does exact recombination close the 0.037 gap to KLSA at a one-second clock?

The 1000-graph author table (``docs/temporary/RLNN_v8_Paper_Table.md``) puts the
captured search at 19.9333 while KLSA reports 19.97 -- a 0.037 gap.  Everything
speed-related is exhausted (700+ steps, each further doubling worth +0.01), so the
only lever left is the quality of the *set of solutions* the search already holds.

v7 measured exact recombination of K=20 legal solutions at **+0.088** under a fixed
workload.  Here the same operator runs inside a real clock: the search stops
``--reserve`` seconds early, the K particle solutions are decoded, every pair is
recombined exactly (Koenig) and polished, and the best candidate is handed to the
frozen PAVE.  Three arms:

* ``captured``  -- the current best arm (full clock, cached decode, then PAVE);
* ``recombine`` -- search to ``deadline - reserve``, then decode + recombine + PAVE;
* ``oracle``    -- full-clock search, recombine computed *after* the clock.

``oracle`` is not deployable; it bounds what the operator could buy if it were free.
If ``captured`` and ``recombine`` are indistinguishable and ``oracle`` is much better,
the operator needs a cheaper implementation; if all three are equal, the particles
have converged and nothing is left inside their union.

    python -u models/eval_v8_recombine.py --split validation --test_samples 100 \
        --budgets 0.5 1.0 --seeds 1017 --reference_seed 0 --reserve 0.05 \
        --result_dir outputs/v8_macro/recombine
"""
import csv
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from diagnose_v7_census import build_parser
from eval_v2 import dataset_at, load_policy, sync, write_run
from eval_v7_budget import DECODE_INTERVAL, POST_RESERVE, run_pave
from eval_v8_capture_budget import paired_bootstrap
from mis_v1 import adjacency
from mis_v3_search import SearchGraph
from mis_v3_value import PAVEV3
from mis_v7_recombine import adjacency_masks, best_recombination, is_independent, polish
from mis_v8_capture import CapturedSearch
from mis_v8_decode import CapturedDecoder
from v7_runs import pave, provenance

ARMS = ("captured", "recombine", "oracle")


def build_recombine_parser():
    parser = build_parser()
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--post_reserve", type=float, default=POST_RESERVE)
    parser.add_argument("--chunk", type=int, default=DECODE_INTERVAL)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--reserve", type=float, default=0.05,
                        help="seconds taken out of the clock for decode + recombination")
    parser.add_argument("--decode_chunk", type=int, default=4)
    parser.add_argument("--shards", type=int, default=1, help="split the graph list into N processes")
    parser.add_argument("--shard", type=int, default=0, help="which shard this process runs")
    parser.set_defaults(horizon=1, num_t=100, split="validation")
    return parser


def masks_of_matrix(solutions):
    """Pack every column of the (nodes, particles) matrix into one bitset."""
    matrix = solutions.to(torch.uint8).cpu().numpy()
    return [int.from_bytes(np.packbits(matrix[:, k], bitorder="little").tobytes(), "little")
            for k in range(matrix.shape[1])]


@torch.no_grad()
def run_arm(solver, graph, adj, normalized, seed, deadline, args, ranker, reserve, recombine):
    """Captured search on this clock, then optionally recombine, then frozen PAVE."""
    device = graph.b.device
    topology = SearchGraph(graph.num_nodes, graph.edge_index, seed=seed)
    neighbours = topology.neighbors
    generator = torch.Generator(device=device).manual_seed(seed)
    search = CapturedSearch(solver, graph, adj, normalized, generator, chunk=DECODE_INTERVAL)
    template = torch.zeros(graph.num_nodes, solver.num_k, device=device, dtype=solver.dtype)
    decoder = CapturedDecoder(solver.problem, adj, graph.b, template, chunk=args.decode_chunk)
    incumbent = 0
    decode_seconds = 0.0

    def decode(best):
        nonlocal incumbent, decode_seconds
        started = perf_counter()
        mask = decoder.decode_best(best)
        decode_seconds += perf_counter() - started
        if mask.bit_count() > incumbent.bit_count():
            incumbent = mask

    decode(search.best_states)
    cutoff = deadline - reserve if recombine else deadline
    steps, extra_seconds, _ = search.run(cutoff, decode, skip_unchanged=True)
    decode_seconds += extra_seconds
    selected = incumbent
    recombine_seconds = 0.0
    gain = 0
    if recombine:
        started = perf_counter()
        masks = masks_of_matrix(decoder.decode(search.best_states))
        candidate, _, _, _ = best_recombination(masks, neighbours, use_polish=True)
        floor = max(selected.bit_count(), polish(selected, neighbours).bit_count())
        gain = candidate.bit_count() - floor
        if candidate.bit_count() > selected.bit_count():
            selected = candidate
        recombine_seconds = perf_counter() - started
    run_pave(topology, selected, ranker, args, deadline)
    sync(device)
    return selected, steps, decode_seconds, recombine_seconds, gain


def main():
    parser = build_recombine_parser()
    args = parser.parse_args()
    if args.chunk != DECODE_INTERVAL:
        parser.error(f"--chunk must match the frozen decode cadence ({DECODE_INTERVAL})")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    output = args.result_dir / "recombine"
    output.mkdir(exist_ok=True)

    solver = load_policy(args)
    solver.model.requires_grad_(False)
    checkpoint = torch.load(pave(args.reference_seed), map_location="cpu", weights_only=False)
    ranker = PAVEV3(checkpoint["mode"], checkpoint.get("stats_dim", 24))
    ranker.load_state_dict(checkpoint["model_state"])
    ranker = ranker.cpu().eval()
    device = solver.accelerator.device
    dataset = dataset_at(args.test_path, solver.dtype, args.test_samples, official=args.split == "test")
    solver.num_t = 20000
    warm = dataset[0].to(device)
    warm_adj = adjacency(warm)
    run_arm(solver, warm, warm_adj, solver.policy_topology(warm, warm_adj), args.seeds[0],
            perf_counter() + 0.25, args, ranker, args.reserve, True)
    sync(device)

    rows = []
    started = perf_counter()
    indices = list(range(len(dataset)))[args.shard::args.shards]
    for position, index in enumerate(indices):
        graph = dataset[index]
        graph_id = f"idx_{index}" if args.split == "test" else Path(dataset.files[index]).stem
        edges = graph.edge_index
        graph = graph.to(device)
        adj = adjacency(graph)
        neighbours = adjacency_masks(graph.num_nodes, edges.cpu().t().tolist())
        for seed in args.seeds:
            for budget in args.budgets:
                normalized = solver.policy_topology(graph, adj)
                order = args.arms if (index + seed) % 2 == 0 else args.arms[::-1]
                for arm in order:
                    sync(device)
                    tick = perf_counter()
                    deadline = tick + budget
                    recombine = arm != "captured"
                    reserve = args.reserve if arm == "recombine" else 0.0
                    selected, steps, decode_seconds, recombine_seconds, gain = run_arm(
                        solver, graph, adj, normalized, seed, deadline, args, ranker,
                        reserve, recombine)
                    seconds = perf_counter() - tick
                    if not is_independent(selected, neighbours):
                        raise ValueError(f"{graph_id}: illegal set in arm {arm}")
                    rows.append(dict(graph_id=graph_id, search_seed=seed, budget=budget, arm=arm,
                                     first=int(order[0] == arm),
                                     final_size=selected.bit_count(),
                                     steps=steps,
                                     recombine_gain=gain,
                                     decode_time_s=round(decode_seconds, 4),
                                     recombine_time_s=round(recombine_seconds, 4),
                                     solve_time_s=round(seconds, 4),
                                     reserve_s=reserve,
                                     deadline_overrun=int(seconds > budget)))
        if (position + 1) % 20 == 0 or position + 1 == len(indices):
            parts = []
            for budget in args.budgets:
                for arm in args.arms:
                    chosen = [r for r in rows if r["budget"] == budget and r["arm"] == arm]
                    if not chosen:
                        continue
                    parts.append(f"{budget}s/{arm}:{mean(r['final_size'] for r in chosen):.3f}"
                                 f"@{mean(r['steps'] for r in chosen):.0f}st"
                                 f"+{mean(r['recombine_gain'] for r in chosen):.2f}")
            print(f"{position + 1}/{len(indices)} " + " ".join(parts), flush=True)

    summary = dict(ref=args.reference_seed, seeds=args.seeds, budgets=args.budgets,
                   graphs=len(indices), horizon=args.horizon, arms=list(args.arms),
                   reserve=args.reserve, decode_chunk=args.decode_chunk,
                   wall_time_s=round(perf_counter() - started, 2))

    def paired(budget, left, right):
        diffs = []
        for graph_id in sorted({r["graph_id"] for r in rows}):
            picked = {}
            for name in (left, right):
                values = [r for r in rows if r["graph_id"] == graph_id
                          and r["budget"] == budget and r["arm"] == name]
                if values:
                    picked[name] = mean(v["final_size"] for v in values)
            if len(picked) == 2:
                diffs.append(picked[left] - picked[right])
        effect, low, high = paired_bootstrap(diffs)
        return dict(left=left, right=right, n=len(diffs), mean=effect, low=low, high=high,
                    wins=sum(d > 0 for d in diffs), ties=sum(d == 0 for d in diffs),
                    losses=sum(d < 0 for d in diffs))

    table = {}
    for budget in args.budgets:
        cell = {}
        for arm in args.arms:
            chosen = [r for r in rows if r["budget"] == budget and r["arm"] == arm]
            cell[arm] = dict(n=len(chosen),
                             final_size=mean(r["final_size"] for r in chosen),
                             steps=mean(r["steps"] for r in chosen),
                             recombine_gain=mean(r["recombine_gain"] for r in chosen),
                             decode_time_s=mean(r["decode_time_s"] for r in chosen),
                             recombine_time_s=mean(r["recombine_time_s"] for r in chosen),
                             overruns=sum(r["deadline_overrun"] for r in chosen))
        cell["paired"] = [paired(budget, arm, "captured") for arm in args.arms if arm != "captured"]
        table[str(budget)] = cell
        print(f"budget={budget}: " + " | ".join(
            f"{arm} {cell[arm]['final_size']:.4f} @ {cell[arm]['steps']:.0f} steps "
            f"(recombine +{cell[arm]['recombine_gain']:.3f})" for arm in args.arms), flush=True)
        for pair in cell["paired"]:
            print(f"  {pair['left']} - {pair['right']}: {pair['mean']:+.4f} "
                  f"[{pair['low']:+.4f},{pair['high']:+.4f}] "
                  f"{pair['wins']}/{pair['ties']}/{pair['losses']}", flush=True)

    summary["table"] = table
    tag = "" if args.shards == 1 else f"_shard{args.shard}"
    summary["shard"] = dict(index=args.shard, count=args.shards)
    metadata = dict(version="v8_recombine", status="completed", mode="recombine",
                    summary=summary, config=vars(args), **provenance(args.reference_seed))
    (output / f"summary_ref{args.reference_seed}{tag}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    path = output / f"recombine_ref{args.reference_seed}{tag}.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_run(output, dict(version="v8_recombine", status="completed", mode="recombine",
                          ref=args.reference_seed, summary=summary, config=vars(args)))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
