"""Put the platform's exchange move *inside* the captured search loop.

The amortised study showed the two components are substitutes: once the search
runs thousands of steps its archive is already a local optimum of the exchange
neighbourhood, so PAVE at the end has nothing to do (platform gain +0.004 at
1 s).  The retrain study showed the frozen objective cannot lift the step
curve.  This harness tests the remaining cheap hypothesis: alternate the two —
search a cycle, decode the archive, run the exchange platform for a slice of the
clock, feed the result back into one particle, keep searching.

Arms (all inside the same clock, all end with the usual final PAVE):

``search``      pure captured search (the baseline of ``eval_v8_padded``)
``inject``      search / exchange cycles, result fed into particle 0 only, so
                the other particles keep their own basins
``refill``      same, but every particle restarts from the exchanged set
"""

import csv
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from diagnose_v7_census import solution_masks
from eval_v2 import dataset_at, load_policy, sync
from eval_v7_budget import run_pave
from eval_v8_capture_budget import paired_bootstrap
from eval_v8_padded import build_parser as build_padded_parser, shape_of
from mis_v1 import adjacency
from mis_v3_search import SearchGraph
from mis_v3_value import PAVEV3
from mis_v7_recombine import is_independent
from mis_v8_pad import PaddedSearch
from v7_runs import pave

ARMS = ("search", "inject", "refill", "kick")


def build_parser():
    parser = build_padded_parser()
    # The base parser pins --arms to the engine names; this harness names its arms
    # after the schedule instead, so the choice list has to go.
    for action in parser._actions:
        if action.dest == "arms":
            action.choices = None
    parser.add_argument("--kick_fraction", type=float, default=0.05,
                        help="share of the incumbent dropped by the perturbation arm")
    parser.add_argument("--cycle_steps", type=int, default=100,
                        help="search steps between two exchange rounds")
    parser.add_argument("--pave_share", type=float, default=0.02,
                        help="clock slice given to each exchange, as a share of the budget")
    parser.set_defaults(arms=list(ARMS), decoders=["cached"])
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    output = args.result_dir / "interleave"
    output.mkdir(exist_ok=True)

    solver = load_policy(args)
    solver.model.requires_grad_(False)
    checkpoint = torch.load(pave(args.reference_seed), map_location="cpu", weights_only=False)
    ranker = PAVEV3(checkpoint["mode"], checkpoint.get("stats_dim", 24))
    ranker.load_state_dict(checkpoint["model_state"])
    ranker = ranker.cpu().eval()
    device = solver.accelerator.device
    dataset = dataset_at(args.test_path, solver.dtype, args.test_samples,
                         official=args.split == "test")
    indices = list(range(len(dataset)))[args.shard::args.shards]
    n_max, capacity = shape_of(dataset, args.padded_shapes)
    print(f"padded shape: nodes {n_max}, capacity {capacity}", flush=True)

    entered = perf_counter()
    engine = PaddedSearch(solver, n_max=n_max, capacity=capacity, chunk=args.chunk,
                          device=device, decode_chunk=args.decode_chunk)
    capture_seconds = perf_counter() - entered
    engine.load(dataset[0].to(device), dataset[0].edge_index, args.seeds[0])
    engine.run(perf_counter() + 0.05, lambda states: None)
    sync(device)

    extended = 20000
    solver.num_t = extended
    rows = []
    started = perf_counter()
    for position, index in enumerate(indices):
        graph = dataset[index]
        graph_id = f"idx_{index}" if args.split == "test" else Path(dataset.files[index]).stem
        edges = graph.edge_index
        n = int(graph.num_nodes)
        graph = graph.to(device)
        adj = adjacency(graph)
        for seed in args.seeds:
            for budget in args.budgets:
                topology = SearchGraph(n, edges, seed=seed)
                neighbours = topology.neighbors
                order = list(args.arms) if (index + seed) % 2 == 0 else list(args.arms)[::-1]
                for repeat in range(args.repeats):
                    arms = order if repeat == 0 else order[::-1]
                    for arm in arms:
                        sync(device)
                        tick = perf_counter()
                        deadline = tick + budget
                        reserve = min(args.post_reserve, args.reserve_frac * budget)
                        cutoff = deadline - reserve
                        incumbent = 0
                        masks_box = []

                        def decode(best):
                            candidate = engine.decode_best(best)
                            if candidate.bit_count() > incumbent:
                                masks_box.append(candidate)

                        loaded = engine.load(graph, edges, seed)
                        decode(engine.best_states)
                        steps = blocks = 0
                        decode_seconds = 0.0
                        exchanges = injections = 0
                        exchange_gain = 0
                        pave_seconds = 0.0
                        max_blocks = max(1, args.cycle_steps // args.chunk)
                        while True:
                            now = perf_counter()
                            if now >= cutoff:
                                break
                            # The baseline gets the whole clock in one run; the
                            # interleaved arms take it in cycles of a fixed number
                            # of steps.
                            cycle_steps, cycle_decode, cycle_blocks = engine.run(
                                cutoff, decode,
                                max_blocks=None if arm == "search" else max_blocks,
                                skip_unchanged=True, decode_every=args.decode_every)
                            steps += cycle_steps
                            blocks += cycle_blocks
                            decode_seconds += cycle_decode
                            if arm == "search" or cycle_steps == 0:
                                break
                            if arm == "kick":
                                # Perturbation instead of an exchange: drop a few
                                # nodes of the incumbent and let the search repair
                                # the hole.  This is the ARW-style escape move.
                                best = max(masks_box, key=int.bit_count) if masks_box else 0
                                chosen = [i for i in range(n) if (best >> i) & 1]
                                if not chosen:
                                    break
                                drop = max(1, int(round(args.kick_fraction * len(chosen))))
                                generator = torch.Generator().manual_seed(
                                    seed + 7919 * (position + 1) + exchanges)
                                picked = torch.randperm(len(chosen), generator=generator)[:drop]
                                kicked = best
                                for offset in picked.tolist():
                                    kicked &= ~(1 << chosen[offset])
                                engine.prime(kicked, n=n, particle=0)
                                exchanges += 1
                                injections += 1
                                continue
                            # Exchange round: give the platform a slice of the clock
                            # and feed the result back into the search.
                            exchange_deadline = min(perf_counter() + args.pave_share * budget,
                                                    cutoff)
                            best = max(masks_box, key=int.bit_count) if masks_box else 0
                            began = perf_counter()
                            exchanged = run_pave(topology, best, ranker, args, exchange_deadline)
                            pave_seconds += perf_counter() - began
                            exchanges += 1
                            if exchanged.bit_count() > best.bit_count():
                                target = None if arm == "refill" else 0
                                engine.prime(exchanged, n=n, particle=target)
                                injections += 1
                                exchange_gain += exchanged.bit_count() - best.bit_count()
                                incumbent = exchanged.bit_count()
                                masks_box.append(exchanged)
                        search_seconds = perf_counter() - tick
                        selected = max(masks_box, key=int.bit_count) if masks_box else 0
                        final = run_pave(topology, selected, ranker, args, deadline)
                        sync(device)
                        seconds = perf_counter() - tick
                        if not is_independent(final, neighbours):
                            raise ValueError(f"{graph_id}: illegal set in {arm}")
                        cell_name = f"{arm}/{args.decoders[0]}"
                        rows.append(dict(graph_id=graph_id, search_seed=seed, budget=budget,
                                         arm=arm, decoder=args.decoders[0], cell=cell_name,
                                         repeat=repeat, first=int(arms[0] == arm),
                                         final_size=final.bit_count(),
                                         search_size=selected.bit_count(),
                                         pave_gain=final.bit_count() - selected.bit_count(),
                                         steps=steps, blocks=blocks,
                                         exchanges=exchanges, injections=injections,
                                         exchange_gain=exchange_gain,
                                         solve_time_s=round(seconds, 4),
                                         search_time_s=round(search_seconds, 4),
                                         decode_time_s=round(decode_seconds, 4),
                                         pave_time_s=round(pave_seconds, 4),
                                         load_s=round(loaded, 4),
                                         reserve_s=round(reserve, 4),
                                         prelude_s=round(loaded, 4),
                                         ms_per_step=round(1e3 * search_seconds / max(steps, 1), 3),
                                         deadline_overrun=int(seconds > budget)))
        if (position + 1) % 20 == 0 or position + 1 == len(indices):
            parts = []
            for budget in args.budgets:
                for arm in args.arms:
                    chosen = [r for r in rows if r["budget"] == budget and r["arm"] == arm]
                    if chosen:
                        parts.append(f"{budget}s/{arm}:"
                                     f"{mean(r['final_size'] for r in chosen):.3f}"
                                     f"@{mean(r['steps'] for r in chosen):.0f}st"
                                     f"/{mean(r['exchanges'] for r in chosen):.1f}x")
            print(f"{position + 1}/{len(indices)} " + " ".join(parts), flush=True)

    columns = list(rows[0].keys())
    with (output / "interleave_ref0.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[name] for name in columns])

    names = list(args.arms)
    summary = dict(ref=args.reference_seed, seeds=args.seeds, budgets=args.budgets,
                   graphs=len(indices), repeats=args.repeats, horizon=args.horizon,
                   post_reserve=args.post_reserve, chunk=args.chunk, arms=names,
                   cycle_steps=args.cycle_steps, pave_share=args.pave_share,
                   decode_every=args.decode_every, decode_chunk=args.decode_chunk,
                   padded=dict(nodes=n_max, capacity=capacity,
                               capture_s=round(capture_seconds, 4)),
                   wall_time_s=round(perf_counter() - started, 2))
    if args.shards > 1:
        summary["shard"] = dict(index=args.shard, count=args.shards)

    for budget in args.budgets:
        for name in names:
            chosen = [r for r in rows if r["budget"] == budget and r["arm"] == name]
            if not chosen:
                continue
            print(f"{budget:>4}s {name:<8} size {mean(r['final_size'] for r in chosen):.4f} "
                  f"search {mean(r['search_size'] for r in chosen):.4f} "
                  f"steps {mean(r['steps'] for r in chosen):.0f} "
                  f"exch {mean(r['exchanges'] for r in chosen):.1f} "
                  f"inject {mean(r['injections'] for r in chosen):.1f} "
                  f"gain {mean(r['exchange_gain'] for r in chosen):.2f} "
                  f"pave {mean(r['pave_time_s'] for r in chosen) * 1e3:.1f} ms", flush=True)

    pairs = {}
    for budget in args.budgets:
        base = [r for r in rows if r["budget"] == budget and r["arm"] == names[0]]
        if not base:
            continue
        by_graph = {}
        for row in base:
            by_graph.setdefault((row["graph_id"], row["search_seed"], row["repeat"]), row)
        for name in names[1:]:
            diffs = []
            for row in rows:
                if row["budget"] != budget or row["arm"] != name:
                    continue
                key = (row["graph_id"], row["search_seed"], row["repeat"])
                if key in by_graph:
                    diffs.append(row["final_size"] - by_graph[key]["final_size"])
            if not diffs:
                continue
            center, low, high = paired_bootstrap(diffs)
            wins = sum(1 for value in diffs if value > 0)
            losses = sum(1 for value in diffs if value < 0)
            ties = len(diffs) - wins - losses
            text = (f"{center:+.4f} [{low:+.4f},{high:+.4f}] "
                    f"{wins}/{ties}/{losses}")
            print(f"{budget:>4}s {name} - {names[0]}: {text}", flush=True)
            pairs[f"{budget}/{name}"] = dict(mean=center, low=low, high=high,
                                             wins=wins, ties=ties, losses=losses,
                                             n=len(diffs))
    summary["paired"] = pairs
    summary["rows"] = len(rows)
    if args.shards == 1:
        with (output / "summary_ref0.json").open("w") as handle:
            json.dump(summary, handle, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "paired"}, indent=2))


if __name__ == "__main__":
    main()
