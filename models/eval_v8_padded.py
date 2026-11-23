"""Evaluate frozen RLNN search and postprocessing under a shared per-graph budget."""

import csv
import json
import math
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from diagnose_v7_census import solution_masks
from eval_v2 import dataset_at, load_policy, sync
from eval_v7_budget import run_pave
from eval_v8_capture_budget import build_capture_parser, paired_bootstrap
from mis_v1 import adjacency
from mis_v3_search import SearchGraph
from mis_v3_value import PAVEV3
from mis_v7_recombine import is_independent, polish, recombine
from mis_v8_pad import PaddedSearch
from mis_v10_lns import LNSController
from v7_runs import pave


def shape_of(dataset, limit=None):
    """Largest node count and largest out-degree in the split."""
    nodes = degree = 0
    for index in range(len(dataset) if limit is None else min(limit, len(dataset))):
        graph = dataset[index]
        edges = graph.edge_index
        nodes = max(nodes, int(graph.num_nodes))
        degree = max(degree, int(torch.bincount(edges[0], minlength=nodes).max()))
    return nodes, degree + 1


def decode_top_best(problem, states, energy, adj, bias, nodes, count):
    """Return the largest independent set among the decoded top-k particles."""
    index = torch.argsort(energy.reshape(-1), dim=0)[: int(count)]
    subset = states[:nodes][:, index].contiguous()
    solutions = problem.decode_gradient_flip(subset, adj, bias, return_solutions=True)
    sizes = solutions.sum(dim=0)
    _, best = torch.max(sizes, dim=0)
    column = solutions[:, int(best)].to(torch.uint8)
    packed = np.packbits(column.cpu().numpy(), bitorder="little")
    return int.from_bytes(packed.tobytes(), "little")


def decode_top_masks(problem, states, energy, adj, bias, nodes, count):
    """A.3 decode the ``count`` lowest-energy particles and keep every mask.

    Same selection as :func:`decode_top_best`, but the whole decoded batch is
    returned so a recombination operator gets a population instead of a point.
    """
    index = torch.argsort(energy.reshape(-1), dim=0)[: int(count)]
    subset = states[:nodes][:, index].contiguous()
    solutions = problem.decode_gradient_flip(subset, adj, bias,
                                            return_solutions=True).to(torch.uint8)
    masks = set()
    for column in range(solutions.shape[1]):
        packed = np.packbits(solutions[:, column].cpu().numpy(), bitorder="little")
        masks.add(int.from_bytes(packed.tobytes(), "little"))
    return sorted(masks, key=int.bit_count, reverse=True)[: int(count)]


def decode_archive(engine, problem, states, adj, bias, nodes, kind, args):
    if args.recombine_top:
        return decode_top_masks(problem, states, engine.best_energy, adj, bias,
                                nodes, args.recombine_top)
    if kind == "eager":
        return solution_masks(problem, states[:nodes], adj, bias)
    if args.decode_top:
        return [decode_top_best(problem, states, engine.best_energy, adj, bias,
                                nodes, args.decode_top)]
    return [engine.decode_best(states)]


def recombine_chain(masks, neighbours, rounds=2):
    """Fold a decoded population into one set by exact pairwise recombination.

    Each fold replaces the incumbent only when Koenig's bound is strictly larger,
    so the chain is monotone in size and never leaves the independent-set family.
    The trailing polish is the A.3 fixed point, which is also monotone.
    """
    if not masks:
        return 0
    ordered = sorted(masks, key=int.bit_count, reverse=True)
    incumbent = ordered[0]
    for _ in range(max(1, int(rounds))):
        grown = False
        for other in ordered[1:]:
            result = recombine(incumbent, other, neighbours)
            if result.size > incumbent.bit_count():
                incumbent = result.candidate
                grown = True
        if not grown:
            break
    return polish(incumbent, neighbours)


def cell_of(arm, decoder, temperature, anneal):
    """Cell name; the deployed temperature keeps the old name so old CSVs match."""
    if float(temperature) == 1.0:
        return f"{arm}/{decoder}"
    suffix = "a" if anneal else ""
    return f"{arm}/{decoder}/T{temperature:g}{suffix}"


def build_parser():
    parser = build_capture_parser()
    parser.add_argument("--reserve_frac", type=float, default=0.2,
                        help="PAVE reserve as a fraction of the budget (capped by --post_reserve)")
    parser.add_argument("--search_backend", choices=("captured", "eager"), default="captured",
                        help="run the same padded search blocks with CUDA Graph replay or eager execution")
    parser.add_argument("--decode_schedule", choices=("midpoint", "terminal"), default="midpoint",
                        help="include or omit the extra mid-search decode")
    parser.add_argument("--decode_every", type=int, default=1,
                        help="decode the archive every N search blocks")
    parser.add_argument("--decode_top", type=int, default=0,
                        help="decode only this many lowest-energy particles "
                             "(0 decodes all of them with the captured decoder)")
    parser.add_argument("--recombine_top", type=int, default=0,
                        help="decode this many particles, then fold them by exact "
                             "recombination (0 disables the operator)")
    parser.add_argument("--recombine_rounds", type=int, default=2,
                        help="passes of the recombination chain")
    parser.add_argument("--debug_cells", action="store_true",
                        help="print per-cell box/pool counts")
    parser.add_argument("--lns_radius", type=int, default=0,
                        help="destroy a radius-r ball between blocks (0 disables v10)")
    parser.add_argument("--lns_period", type=int, default=2,
                        help="block period of the destroy/repair cycle")
    parser.add_argument("--lns_particle", type=int, default=0,
                        help="archive column whose selected set seeds the ball")
    parser.add_argument("--lns_no_restore", action="store_true",
                        help="keep the repaired state instead of restoring the archive")
    parser.add_argument("--dense", action="store_true",
                        help="hand the policy a dense adjacency instead of CSR")
    parser.add_argument("--reserve_ms", type=float, default=0.0,
                        help="v13/t21: explicit reserve held out of the search clock in "
                             "milliseconds (0 keeps the derived reserve). Needed for PAVE-off "
                             "arms, which otherwise spend the whole budget and can overrun by "
                             "the jitter of one 10-step block")
    parser.add_argument("--compile_model", action="store_true",
                        help="v13 P1: torch.compile the frozen policy before the engine "
                             "is built (kernel fusion; default off keeps A0 bit-identical)")
    parser.add_argument("--fixed_steps", type=int, default=0,
                        help="v11 gate 0: run exactly this many steps and ignore the "
                             "clock cutoff (0 keeps the deadline-driven loop)")
    parser.add_argument("--skip_incumbent", action="store_true",
                        help="v13 P0.5: skip the incumbent decode that runs before the "
                             "search loop.  It costs 45-53 ms/graph at 1000 graphs "
                             "(26-28%% of a 0.2 s budget) and the terminal decode of "
                             "engine.run already supplies the mask; a 200-graph x 130-step "
                             "check found search_size identical (models/probe_p05_budget.py)")
    parser.add_argument("--no_pave", action="store_true",
                        help="v13 P0.5: skip the PAVE post-pass entirely.  final_size "
                             "then equals search_size and the whole budget goes to the "
                             "search loop (reserve forced to 0).  Reported as pave=false; "
                             "never mix these rows into a PAVE arm's mean")
    parser.add_argument("--v11_csv", default=None,
                        help="v11: also write the per-graph CSV with the v11 column "
                             "schema (graph_id,budget,cell,seed,repeat,search_size,"
                             "final_size,steps,wall_ms,...); the legacy outputs are "
                             "left untouched")
    parser.add_argument("--padded_shapes", type=int, default=None,
                        help="cap the shape scan (default: the whole split)")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1.0],
                        help="policy temperatures; 1.0 is the deployed policy")
    parser.add_argument("--anneal", action="store_true",
                        help="cool each temperature back to 1.0 along the budget")
    parser.set_defaults(arms=["padded"], decoders=["captured", "cached"])
    return parser


def archive_backend(kind, args):
    return "eager" if kind == "eager" or args.decode_top or args.recombine_top else "captured"


def archive_particles(kind, args):
    count = args.recombine_top or (args.decode_top if kind != "eager" else 0)
    return min(count or args.num_k, args.num_k)


def load_ranker(reference_seed):
    checkpoint = torch.load(pave(reference_seed), map_location="cpu", weights_only=False)
    ranker = PAVEV3(checkpoint["mode"], checkpoint.get("stats_dim", 24))
    ranker.load_state_dict(checkpoint["model_state"])
    return ranker.cpu().eval()


def run_cell(engine, graph, edges, adj, topology, ranker, decode_cost, args, *,
             graph_id, seed, budget, repeat, arm, decoder_kind, temp, first):
    device = engine.device
    n = int(graph.num_nodes)
    neighbours = topology.neighbors
    sync(device)
    tick = perf_counter()
    deadline = tick + budget
    reserve = (0.0 if args.no_pave
               else min(args.post_reserve, args.reserve_frac * budget))
    if args.reserve_ms:
        reserve = 1e-3 * args.reserve_ms
    cutoff = deadline - reserve
    fixed_blocks = None
    if args.fixed_steps:
        cutoff = float("inf")
        fixed_blocks = math.ceil(args.fixed_steps / args.chunk)
    selected = 0
    search_ready = None
    last_decode_ready = None
    pool = []

    def decode(best, kind=decoder_kind):
        nonlocal selected, search_ready, last_decode_ready
        decode_started = perf_counter()
        if fixed_blocks is None and decode_started >= deadline:
            return
        masks = decode_archive(
            engine, engine.problem, best, adj, graph.b, n, kind, args)
        candidate = max(masks, key=int.bit_count)
        decoded = perf_counter()
        last_decode_ready = decoded
        if fixed_blocks is None:
            decode_cost[kind] = decoded - decode_started
            if decoded > deadline:
                return
        if args.recombine_top:
            for mask in masks:
                if mask not in pool:
                    pool.append(mask)
        if search_ready is None or candidate.bit_count() > selected.bit_count():
            selected = candidate
            search_ready = decoded

    decode_reserve = (None if fixed_blocks is not None
                      else max(0.005, 2 * decode_cost[decoder_kind]))
    loaded = engine.load(graph, edges, seed)
    if not args.skip_incumbent:
        decode(engine.best_states)
    prelude = perf_counter() - tick
    controller = None
    if args.lns_radius:
        controller = LNSController(engine, neighbours,
                                   radius=args.lns_radius,
                                   period=args.lns_period,
                                   particle=args.lns_particle,
                                   restore=not args.lns_no_restore,
                                   seed=seed)
    steps, decode_seconds, blocks = engine.run(
        cutoff, decode, skip_unchanged=decoder_kind == "cached",
        decode_every=args.decode_every, temperature=temp,
        anneal=args.anneal,
        on_block=None if controller is None else controller.on_block,
        max_blocks=fixed_blocks, decode_reserve=decode_reserve,
        midpoint_decode=args.decode_schedule == "midpoint")
    sync(device)
    search_seconds = perf_counter() - tick
    pave_deadline = (deadline if fixed_blocks is None
                     else perf_counter() + reserve)
    if args.debug_cells:
        print(f"  debug {graph_id} decoded {search_ready is not None} "
              f"selected {selected.bit_count()} pool {len(pool)}",
              flush=True)
    final = selected
    final_ready = search_ready
    if not args.no_pave and perf_counter() < pave_deadline:
        candidate = run_pave(topology, selected, ranker, args, pave_deadline)
        finished = perf_counter()
        if fixed_blocks is not None or finished <= deadline:
            final = candidate
            final_ready = finished
    recombine_size = 0
    recombine_seconds = 0.0
    if args.recombine_top and len(pool) >= 2 and (
        fixed_blocks is not None or perf_counter() < deadline
    ):
        folded = perf_counter()
        combined = recombine_chain(pool, neighbours,
                                   args.recombine_rounds)
        recombine_seconds = perf_counter() - folded
        recombine_size = combined.bit_count()
        if recombine_size > final.bit_count() and (
            fixed_blocks is not None or folded + recombine_seconds <= deadline
        ):
            if not is_independent(combined, neighbours):
                raise ValueError(
                    f"{graph_id}: illegal recombined set")
            final = combined
            final_ready = folded + recombine_seconds
    seconds = perf_counter() - tick
    if not is_independent(final, neighbours):
        raise ValueError(f"{graph_id}: illegal set in {arm}/{decoder_kind}")
    cell_name = cell_of(arm, decoder_kind, temp, args.anneal)
    return dict(graph_id=graph_id, search_seed=seed, budget=budget,
                 arm=arm, decoder=decoder_kind, cell=cell_name,
                 repeat=repeat,
                 first=int(first),
                 temperature=temp, anneal=int(args.anneal),
                 final_size=final.bit_count(),
                 search_size=selected.bit_count(),
                 pave_gain=final.bit_count() - selected.bit_count(),
                 recombine_size=recombine_size,
                 recombine_pool=len(pool),
                 recombine_s=round(recombine_seconds, 4),
                 lns_rounds=0 if controller is None else controller.rounds,
                 lns_evicted=0 if controller is None else controller.evicted,
                 steps=steps, blocks=blocks,
                 solve_time_s=round(seconds, 4),
                 search_time_s=round(search_seconds, 4),
                 decode_time_s=round(decode_seconds, 4),
                 last_decode_ready_s=(None if last_decode_ready is None
                                 else last_decode_ready - tick),
                 search_ready_s=(None if search_ready is None
                                 else search_ready - tick),
                 final_ready_s=(None if final_ready is None
                                 else final_ready - tick),
                 decode_reserve_s=decode_reserve,
                 load_s=round(loaded, 4),
                 reserve_s=round(reserve, 4),
                 prelude_s=round(prelude, 4),
                 ms_per_step=round(1e3 * search_seconds / max(steps, 1), 3),
                 deadline_overrun=int(seconds > budget),
                 decode_backend=archive_backend(decoder_kind, args),
                 decode_particles=archive_particles(decoder_kind, args))


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    output = args.result_dir / "padded"
    output.mkdir(exist_ok=True)

    solver = load_policy(args)
    solver.model.requires_grad_(False)
    ranker = None if args.no_pave else load_ranker(args.reference_seed)
    device = solver.accelerator.device
    dataset = dataset_at(args.test_path, solver.dtype, args.test_samples,
                         official=args.split == "test")
    indices = list(range(len(dataset)))[args.shard::args.shards]
    n_max, capacity = shape_of(dataset, args.padded_shapes)
    if not indices:
        parser.error("the requested split and shard contain no graphs")
    print(f"padded shape: nodes {n_max}, capacity {capacity}", flush=True)

    setup_started = perf_counter()
    if args.compile_model:
        solver.model = torch.compile(solver.model, mode="default", fullgraph=False,
                                     dynamic=False)
        print(f"compiled policy: {type(solver.model).__name__}", flush=True)

    entered = perf_counter()
    capture_decode = any(archive_backend(kind, args) == "captured" for kind in args.decoders)
    engine = PaddedSearch(solver, n_max=n_max, capacity=capacity, chunk=args.chunk, device=device,
                          decode_chunk=args.decode_chunk if capture_decode else 0, dense=args.dense,
                          capture=args.search_backend == "captured")
    capture_seconds = perf_counter() - entered
    warm_graph = dataset[0]
    warm_edges = warm_graph.edge_index
    warm_graph = warm_graph.to(device)
    engine.load(warm_graph, warm_edges, args.seeds[0])
    engine.run(perf_counter() + 0.05, lambda states: None)
    sync(device)
    decode_cost = {}
    if not args.fixed_steps:
        warm_adj = adjacency(warm_graph)
        for kind in dict.fromkeys(args.decoders):
            for _ in range(2):
                warmed = perf_counter()
                decode_archive(engine, solver.problem, engine.best_states, warm_adj,
                               warm_graph.b, int(warm_graph.num_nodes), kind, args)
                sync(device)
                decode_cost[kind] = perf_counter() - warmed

    extended = 20000
    solver.num_t = extended
    cells = [(arm, kind, temp) for arm in args.arms for kind in args.decoders
             for temp in args.temperatures]
    rows = []
    started = perf_counter()
    for position, index in enumerate(indices):
        graph = dataset[index]
        graph_id = f"idx_{index}" if args.split == "test" else Path(dataset.files[index]).stem
        edges = graph.edge_index
        graph = graph.to(device)
        adj = adjacency(graph)
        for seed in args.seeds:
            for budget in args.budgets:
                topology = SearchGraph(int(graph.num_nodes), edges, seed=seed)
                order = cells if (index + seed) % 2 == 0 else cells[::-1]
                for repeat in range(args.repeats):
                    arms = order if repeat == 0 else order[::-1]
                    for arm, decoder_kind, temp in arms:
                        rows.append(run_cell(
                            engine, graph, edges, adj, topology, ranker, decode_cost, args,
                            graph_id=graph_id, seed=seed, budget=budget, repeat=repeat,
                            arm=arm, decoder_kind=decoder_kind, temp=temp,
                            first=arms[0][:2] == (arm, decoder_kind)))
        if (position + 1) % 20 == 0 or position + 1 == len(indices):
            parts = []
            for budget in args.budgets:
                for arm, kind, temp in cells:
                    name = cell_of(arm, kind, temp, args.anneal)
                    chosen = [r for r in rows if r["budget"] == budget
                              and r["cell"] == name]
                    if chosen:
                        parts.append(f"{budget}s/{name}:"
                                     f"{mean(r['final_size'] for r in chosen):.3f}"
                                     f"@{mean(r['steps'] for r in chosen):.0f}st")
            print(f"{position + 1}/{len(indices)} " + " ".join(parts), flush=True)

    dtype_tag = {torch.float16: "fp16", torch.float32: "fp32",
                 torch.bfloat16: "bf16"}.get(solver.dtype, str(solver.dtype))
    dense_flag = int(engine.use_dense)
    config = dict(search_backend=args.search_backend, decode_schedule=args.decode_schedule,
                  dtype=dtype_tag, dense=dense_flag, num_k=args.num_k,
                  decode_every=args.decode_every, decode_top=args.decode_top,
                  reserve_frac=args.reserve_frac)
    names = [cell_of(arm, kind, temp, args.anneal) for arm, kind, temp in cells]
    summary = dict(ref=args.reference_seed, seeds=args.seeds, budgets=args.budgets,
                   graphs=len(indices), repeats=args.repeats, horizon=args.horizon,
                   post_reserve=args.post_reserve, chunk=args.chunk, arms=list(args.arms),
                   decoders=list(args.decoders), decode_chunk=args.decode_chunk, cells=names,
                   temperatures=list(args.temperatures), anneal=bool(args.anneal),
                   leading_decode="off" if args.skip_incumbent else "on",
                   pave=not args.no_pave, fixed_steps=args.fixed_steps,
                   compile_model=args.compile_model, setup_s=round(started - setup_started, 4),
                   test_path=str(args.test_path), checkpoint_dir=str(args.save_dir),
                   budgeted_decode=not bool(args.fixed_steps),
                   reserve_ms=float(args.reserve_ms),
                   padded=dict(nodes=n_max, capacity=capacity,
                               capture_s=round(capture_seconds, 4),
                               load_s=round(mean(r["load_s"] for r in rows), 4)),
                   decode_config={kind: dict(backend=archive_backend(kind, args),
                                             particles=archive_particles(kind, args))
                                  for kind in args.decoders},
                   wall_time_s=round(perf_counter() - started, 2), **config)
    if args.shards > 1:
        summary["shard"] = dict(index=args.shard, count=args.shards)

    for budget in args.budgets:
        for name in names:
            chosen = [r for r in rows if r["budget"] == budget and r["cell"] == name]
            if not chosen:
                continue
            print(f"{budget:>4}s {name:<18} size {mean(r['final_size'] for r in chosen):.4f} "
                  f"steps {mean(r['steps'] for r in chosen):.1f} "
                  f"search {mean(r['search_size'] for r in chosen):.4f} "
                  f"ms/step {mean(r['ms_per_step'] for r in chosen):.3f} "
                  f"load {mean(r['load_s'] for r in chosen) * 1e3:.2f} ms "
                  f"decode {mean(r['decode_time_s'] for r in chosen) * 1e3:.2f} ms", flush=True)

    for budget in args.budgets:
        base = cell_of(args.arms[0], args.decoders[0], 1.0, args.anneal)
        for name in names[1:]:
            diffs = []
            for graph_id in sorted({r["graph_id"] for r in rows}):
                left = [r for r in rows if r["graph_id"] == graph_id and r["budget"] == budget
                        and r["cell"] == name]
                right = [r for r in rows if r["graph_id"] == graph_id and r["budget"] == budget
                         and r["cell"] == base]
                if left and right:
                    diffs.append(mean(r["final_size"] for r in left)
                                 - mean(r["final_size"] for r in right))
            if diffs:
                gap, low, high = paired_bootstrap(diffs)
                print(f"{budget}s {name} - {base}: {gap:+.4f} [{low:+.4f},{high:+.4f}]", flush=True)

    tag = "" if args.shards == 1 else f"_shard{args.shard}"
    path = output / f"padded_ref{args.reference_seed}{tag}"
    columns = list(rows[0])
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[key] for key in columns])
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {path.with_suffix('.csv')}", flush=True)
    if args.v11_csv:
        fp32_csr = int(dtype_tag != "fp16" and not dense_flag)
        columns_v11 = ["graph_id", "budget", "cell", "arm", "decoder", "seed",
                       "repeat", "search_size", "final_size", "steps", "wall_ms",
                       "ms_per_step", "search_ms", "load_ms", "blocks", "pave_gain",
                       "decode_ms", "decode_reserve_ms", "last_decode_ready_ms",
                       "search_ready_ms", "final_ready_ms",
                       "deadline_overrun", "temperature", "anneal", "fixed_steps",
                       "dtype", "dense", "fp32_csr", "skip_incumbent", "no_pave",
                       "compile_model", "search_backend", "decode_schedule",
                       "decode_backend", "decode_particles", "num_k", "decode_every", "decode_top",
                       "reserve_ms"]
        v11_path = Path(args.v11_csv)
        v11_path.parent.mkdir(parents=True, exist_ok=True)
        with v11_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns_v11,
                                    extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row, seed=row["search_seed"],
                                     wall_ms=round(1e3 * row["solve_time_s"], 3),
                                     search_ms=round(1e3 * row["search_time_s"], 3),
                                     load_ms=round(1e3 * row["load_s"], 3),
                                     decode_ms=1e3 * row["decode_time_s"],
                                     decode_reserve_ms=(None if row["decode_reserve_s"] is None
                                                        else 1e3 * row["decode_reserve_s"]),
                                     last_decode_ready_ms=(None if row["last_decode_ready_s"] is None
                                                           else 1e3 * row["last_decode_ready_s"]),
                                     search_ready_ms=(None if row["search_ready_s"] is None
                                                      else 1e3 * row["search_ready_s"]),
                                     final_ready_ms=(None if row["final_ready_s"] is None
                                                     else 1e3 * row["final_ready_s"]),
                                     fixed_steps=args.fixed_steps,
                                     compile_model=int(args.compile_model),
                                     **config,
                                     reserve_ms=1e3 * row["reserve_s"],
                                     fp32_csr=fp32_csr,
                                     skip_incumbent=int(bool(args.skip_incumbent)),
                                     no_pave=int(bool(args.no_pave))))
        print(f"wrote {v11_path}", flush=True)


if __name__ == "__main__":
    main()
