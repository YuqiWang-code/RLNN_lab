"""Equal-budget comparison: eager frozen search against captured block replay.

Same 100 validation graphs, same search seed, same frozen backbone, same A.3
decode cadence, same frozen PAVE (horizon=1), same 0.5 s / 1.0 s clocks.  The only
difference is the engine: ``deadline_search`` steps the frozen policy one step at a
time (2-3 ms/step on this machine), ``captured_search`` replays 10-step CUDA-graph
blocks (0.26 ms/step plus a one-time ~60 ms capture charged inside the clock).
Both arms fill the same clock, so the question is whether the extra step rate buys
size -- the only lever left after nine method lines came back at zero.

    python -u models/eval_v8_capture_budget.py --split validation --test_samples 100 \
        --budgets 0.5 1.0 --seeds 1017 --reference_seed 0 \
        --result_dir outputs/v8_macro/positioning
"""
import csv
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from diagnose_v7_census import build_parser, solution_masks
from eval_v2 import dataset_at, load_policy, sync, write_run
from eval_v7_budget import DECODE_INTERVAL, POST_RESERVE, deadline_search, run_pave
from mis_v1 import adjacency
from mis_v3_search import SearchGraph
from mis_v3_value import PAVEV3
from mis_v7_recombine import is_independent
from mis_v8_capture import CapturedSearch
from mis_v8_decode import CapturedDecoder
from v7_runs import pave, provenance

ARMS = ("eager", "captured")
DECODERS = ("eager", "captured", "cached")


def build_capture_parser():
    parser = build_parser()
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--post_reserve", type=float, default=POST_RESERVE)
    parser.add_argument("--chunk", type=int, default=DECODE_INTERVAL)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--decoders", nargs="+", choices=DECODERS, default=["eager"])
    parser.add_argument("--decode_chunk", type=int, default=4)
    parser.set_defaults(horizon=1, num_t=100)
    return parser


@torch.no_grad()
def captured_search(solver, graph, adj, seed, cutoff, normalized=None, masks_of=None,
                    skip_unchanged=False):
    """``deadline_search`` with the step replaced by a captured block replay.

    Returns the same tuple: (incumbent, masks, steps, decode_seconds, prelude).
    ``prelude`` includes the capture, so the one-time cost stays inside the clock.
    """
    entered = perf_counter()
    device = graph.b.device
    problem = solver.problem
    if normalized is None:
        normalized = solver.policy_topology(graph, adj)
    generator = torch.Generator(device=device).manual_seed(seed)
    search = CapturedSearch(solver, graph, adj, normalized, generator, chunk=DECODE_INTERVAL)
    incumbent = 0
    masks = None
    decode_seconds = 0.0

    def decode(best):
        nonlocal incumbent, masks, decode_seconds
        started = perf_counter()
        if masks_of is None:
            masks = solution_masks(problem, best, adj, graph.b)
            candidate = max(masks, key=int.bit_count)
        else:
            candidate = masks_of(best)
            masks = [candidate]
        decode_seconds += perf_counter() - started
        if candidate.bit_count() > incumbent.bit_count():
            incumbent = candidate

    decode(search.best_states)
    prelude = perf_counter() - entered
    steps, decode_seconds, _ = search.run(cutoff, decode, skip_unchanged=skip_unchanged)
    return incumbent, masks, steps, decode_seconds, prelude


def paired_bootstrap(diffs, resamples=5000, seed=0):
    if not diffs:
        return 0.0, 0.0, 0.0
    values = torch.tensor(diffs, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    index = torch.randint(0, values.numel(), (resamples, values.numel()), generator=generator)
    means = values[index].mean(dim=1)
    return float(values.mean()), float(means.quantile(0.025)), float(means.quantile(0.975))


def main():
    parser = build_capture_parser()
    args = parser.parse_args()
    if args.chunk != DECODE_INTERVAL:
        parser.error(f"--chunk must match the frozen decode cadence ({DECODE_INTERVAL})")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    output = args.result_dir / "capture_budget"
    output.mkdir(exist_ok=True)

    solver = load_policy(args)
    solver.model.requires_grad_(False)
    checkpoint = torch.load(pave(args.reference_seed), map_location="cpu", weights_only=False)
    ranker = PAVEV3(checkpoint["mode"], checkpoint.get("stats_dim", 24))
    ranker.load_state_dict(checkpoint["model_state"])
    ranker = ranker.cpu().eval()
    device = solver.accelerator.device
    dataset = dataset_at(args.test_path, solver.dtype, args.test_samples, official=args.split == "test")
    extended = 20000
    # Warm the frozen kernels, the A.3 decode, PAVE and one capture so the first
    # measured arm does not spend its whole clock compiling.
    solver.num_t = extended
    warm = dataset[0].to(device)
    warm_adj = adjacency(warm)
    warm_topology = SearchGraph(warm.num_nodes, dataset[0].edge_index, seed=args.seeds[0])
    warm_selected, warm_masks, _, _, _ = deadline_search(solver, warm, warm_adj, args.seeds[0],
                                                         perf_counter() + 0.2)
    run_pave(warm_topology, warm_selected, ranker, args, perf_counter() + 0.1)
    captured_search(solver, warm, warm_adj, args.seeds[0], perf_counter() + 0.2)
    warm_template = torch.zeros(warm.num_nodes, solver.num_k, device=device, dtype=solver.dtype)
    CapturedDecoder(solver.problem, warm_adj, warm.b, warm_template,
                    chunk=args.decode_chunk).decode_best(warm_template)
    sync(device)

    rows = []
    # ``cached`` needs the archive counter that only the captured block keeps.
    cells = [(arm, kind) for arm in args.arms for kind in args.decoders
             if not (arm == "eager" and kind == "cached")]
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
                normalized = solver.policy_topology(graph, adj)
                solver.num_t = extended
                # Throwaway pass of the same shape so both arms meet a warm allocator.
                deadline_search(solver, graph, adj, seed, perf_counter() + 0.25, normalized)
                sync(device)
                order = cells if (index + seed) % 2 == 0 else cells[::-1]
                for repeat in range(args.repeats):
                    arms = order if repeat == 0 else order[::-1]
                    for arm, decoder_kind in arms:
                        engine = deadline_search if arm == "eager" else captured_search
                        sync(device)
                        tick = perf_counter()
                        deadline = tick + budget
                        cutoff = deadline - args.post_reserve
                        masks_of = None
                        skip_unchanged = decoder_kind == "cached"
                        if decoder_kind in ("captured", "cached"):
                            # Charged inside the clock, exactly like the search capture.
                            template = torch.zeros(graph.num_nodes, solver.num_k,
                                                   device=device, dtype=solver.dtype)
                            masks_of = CapturedDecoder(solver.problem, adj, graph.b, template,
                                                       chunk=args.decode_chunk).decode_best
                        if arm == "eager":
                            selected, masks, steps, decode_seconds, prelude = engine(
                                solver, graph, adj, seed, cutoff, normalized, masks_of)
                        else:
                            selected, masks, steps, decode_seconds, prelude = engine(
                                solver, graph, adj, seed, cutoff, normalized, masks_of,
                                skip_unchanged)
                        sync(device)
                        search_seconds = perf_counter() - tick
                        final = run_pave(topology, selected, ranker, args, deadline)
                        sync(device)
                        seconds = perf_counter() - tick
                        if not is_independent(final, neighbours):
                            raise ValueError(f"{graph_id}: illegal set in {arm}/{decoder_kind}")
                        cell_name = f"{arm}/{decoder_kind}"
                        rows.append(dict(graph_id=graph_id, search_seed=seed, budget=budget, arm=arm,
                                         decoder=decoder_kind, cell=cell_name,
                                         repeat=repeat, first=int(arms[0] == (arm, decoder_kind)),
                                         final_size=final.bit_count(),
                                         search_size=selected.bit_count(),
                                         pave_gain=final.bit_count() - selected.bit_count(),
                                         steps=steps,
                                         solve_time_s=round(seconds, 4),
                                         search_time_s=round(search_seconds, 4),
                                         decode_time_s=round(decode_seconds, 4),
                                         prelude_s=round(prelude, 4),
                                         ms_per_step=round(1e3 * search_seconds / max(steps, 1), 3),
                                         deadline_overrun=int(seconds > budget)))
        if (index + 1) % 20 == 0 or index + 1 == len(dataset):
            parts = []
            for budget in args.budgets:
                for arm, kind in cells:
                    chosen = [r for r in rows if r["budget"] == budget and r["cell"] == f"{arm}/{kind}"]
                    if not chosen:
                        continue
                    parts.append(f"{budget}s/{arm}/{kind}:{mean(r['final_size'] for r in chosen):.3f}"
                                 f"@{mean(r['steps'] for r in chosen):.0f}st")
            print(f"{index + 1}/{len(dataset)} " + " ".join(parts), flush=True)

    names = [f"{arm}/{kind}" for arm, kind in cells]
    summary = dict(ref=args.reference_seed, seeds=args.seeds, budgets=args.budgets,
                   graphs=len(dataset), repeats=args.repeats, horizon=args.horizon,
                   post_reserve=args.post_reserve, chunk=args.chunk, arms=list(args.arms),
                   decoders=list(args.decoders), decode_chunk=args.decode_chunk, cells=names,
                   wall_time_s=round(perf_counter() - started, 2))

    def paired(budget, left, right):
        diffs, ratios = [], []
        for graph_id in sorted({r["graph_id"] for r in rows}):
            picked = {}
            for name in (left, right):
                values = [r for r in rows if r["graph_id"] == graph_id
                          and r["budget"] == budget and r["cell"] == name]
                if values:
                    picked[name] = (mean(v["final_size"] for v in values),
                                    mean(v["steps"] for v in values))
            if len(picked) == 2:
                diffs.append(picked[left][0] - picked[right][0])
                ratios.append(picked[left][1] / max(picked[right][1], 1e-9))
        effect, low, high = paired_bootstrap(diffs)
        return dict(left=left, right=right, n=len(diffs), mean=effect, low=low, high=high,
                    wins=sum(d > 0 for d in diffs), ties=sum(d == 0 for d in diffs),
                    losses=sum(d < 0 for d in diffs),
                    step_ratio=mean(ratios) if ratios else 0.0)

    table = {}
    for budget in args.budgets:
        cell = {}
        for name in names:
            chosen = [r for r in rows if r["budget"] == budget and r["cell"] == name]
            cell[name] = dict(n=len(chosen),
                              final_size=mean(r["final_size"] for r in chosen),
                              search_size=mean(r["search_size"] for r in chosen),
                              steps=mean(r["steps"] for r in chosen),
                              ms_per_step=mean(r["ms_per_step"] for r in chosen),
                              prelude_s=mean(r["prelude_s"] for r in chosen),
                              search_time_s=mean(r["search_time_s"] for r in chosen),
                              decode_time_s=mean(r["decode_time_s"] for r in chosen),
                              overruns=sum(r["deadline_overrun"] for r in chosen))
        # Every cell is compared against the slow reference, paired per graph.
        pairs = [paired(budget, name, "eager/eager") for name in names if name != "eager/eager"]
        cell["paired"] = pairs
        table[str(budget)] = cell
        print(f"budget={budget}: " + " | ".join(
            f"{name} {cell[name]['final_size']:.4f} @ {cell[name]['steps']:.0f} steps"
            for name in names), flush=True)
        for pair in pairs:
            print(f"  {pair['left']} - {pair['right']}: {pair['mean']:+.4f} "
                  f"[{pair['low']:+.4f},{pair['high']:+.4f}] "
                  f"{pair['wins']}/{pair['ties']}/{pair['losses']} steps x{pair['step_ratio']:.2f}",
                  flush=True)

    summary["table"] = table
    metadata = dict(version="v8_positioning", status="completed", mode="capture_budget",
                    summary=summary, config=vars(args), **provenance(args.reference_seed))
    (output / f"summary_ref{args.reference_seed}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    path = output / f"capture_budget_ref{args.reference_seed}.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_run(output, dict(version="v8_positioning", status="completed", mode="capture_budget",
                          ref=args.reference_seed, summary=summary, config=vars(args)))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
