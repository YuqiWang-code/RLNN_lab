"""R2-E2: the parallel-arm batch multiplier ``c(N)`` and the N-arm ensemble arm.

R2 (variance route) lives or dies on one quantity: when ``N`` arms of the *same*
graph share one batched forward, how much does a step cost relative to a single
arm?  E1 (t14) fixed the variance side -- three full-budget seeds give a max-of-3
of ``+0.093`` -- and the budget side: splitting 808 steps into ``808/N`` costs
``-0.031`` (N=2) / ``-0.056`` (N=3).  So the arm breaks even only when

    N=2: c <= 1.60   (loose 1.79)      N=3: c <= 1.99   (loose 2.36)

Two modes, both on the frozen padded engine (``models/mis_v8_pad.py`` is on the
frozen list; nothing here edits it -- the engine is used as-is and the arms are
extra particle columns):

``bench``   E2-A: same graph, ``state`` widened to ``num_k = N*K`` columns, N in
            1..4, 200 steps each, median ms/step + a per-component breakdown
            (forward / energy / sampling / decode / rest) + peak memory.
``compare`` E2-B: 200 graphs, one session, single arm (810 steps, A0 flags) vs the
            parallel arm (N arms, ``floor(810/c)`` steps each, per-graph max),
            with per-graph CSV, paired bootstrap CI and a feasibility assertion
            on every returned set.

Arm independence: arms occupy consecutive column blocks and are initialised from
different torch seeds (1017/1019/1021, the project's eval seeds).  The noise
stream inside the captured block is a single ``torch.rand`` over all columns, so
an arm is *not* bit-identical to a standalone run of its seed -- column-level
noise is independent, the stream identity is not.  The parallel arm is therefore
the deployable statistic (N arms in one batch), not a re-derivation of E1's
max-over-three-standalone-runs; ``compare`` prints the per-arm sizes so the gap
between the two statistics is visible.

    python models/mis_v12_parallel.py bench   --arms 1 2 3 4 --steps 200
    python models/mis_v12_parallel.py compare --graphs 200 --c2 1.3 --c3 1.5
    python models/mis_v12_parallel.py compare --from_csv .../compare.csv --c2 2.389
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch

from diagnose_v7_census import build_parser as build_base_parser, solution_masks
from eval_v2 import dataset_at, load_policy, sync
from eval_v7_budget import run_pave
from eval_v8_capture_budget import paired_bootstrap
from eval_v8_padded import build_parser, decode_top_best, shape_of
from mis_v1 import adjacency
from mis_v3_search import SearchGraph
from mis_v3_value import PAVEV3
from mis_v7_recombine import is_independent
from mis_v8_pad import PaddedSearch
from v7_runs import pave

BASE_K = 200
BASE_STEPS = 810
ARM_SEEDS = (1017, 1019, 1021, 1023)


# --------------------------------------------------------------------------- #
# shared plumbing
# --------------------------------------------------------------------------- #
def parsed(argv):
    parser = build_parser()
    parser.add_argument("--bench_arms", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="bench: arm counts to time")
    parser.add_argument("--bench_steps", type=int, default=200,
                        help="bench: steps per arm count")
    parser.add_argument("--graphs", type=int, default=200,
                        help="compare: graphs, taken in dataset order (no cherry-picking)")
    parser.add_argument("--c2", type=float, default=None, help="compare: measured c(2)")
    parser.add_argument("--c3", type=float, default=None, help="compare: measured c(3)")
    parser.add_argument("--decode_count", type=int, default=4,
                        help="A.3 decode this many lowest-energy particles (A0 uses 4)")
    parser.add_argument("--from_csv", default=None,
                        help="compare: recompute summary from an existing compare.csv")
    parser.add_argument("--out", default="outputs/v11/r2/e2_parallel",
                        help="output directory for the CSV/JSON artifacts")
    parser.set_defaults(budgets=[1.0], seeds=[1017], repeats=1, shards=1, shard=0,
                        decoders=["captured"])
    return parser.parse_args(argv)


def setup(args):
    """Solver + ranker + dataset, exactly the hooks ``eval_v8_padded`` uses."""
    solver = load_policy(args)
    solver.model.requires_grad_(False)
    checkpoint = torch.load(pave(args.reference_seed), map_location="cpu", weights_only=False)
    ranker = PAVEV3(checkpoint["mode"], checkpoint.get("stats_dim", 24))
    ranker.load_state_dict(checkpoint["model_state"])
    ranker = ranker.cpu().eval()
    dataset = dataset_at(args.test_path, solver.dtype, args.test_samples,
                         official=args.split == "test")
    solver.num_t = 20000
    return solver, ranker, dataset


def engine_for(solver, n_arms, n_max, capacity, args, device):
    """A padded engine whose particle columns hold ``n_arms`` blocks of K."""
    solver.num_k = BASE_K * n_arms
    engine = PaddedSearch(solver, n_max=n_max, capacity=capacity, chunk=args.chunk,
                          device=device, decode_chunk=args.decode_chunk, dense=args.dense)
    return engine


def arms_from(engine, n_arms, seed_of=None):
    """Re-initialise each K-column block from its own torch seed (after load())."""
    k = engine.num_k // n_arms
    for block in range(n_arms):
        seed = (seed_of or (lambda i: ARM_SEEDS[i]))(block)
        generator = torch.Generator(device=engine.device).manual_seed(int(seed))
        columns = slice(block * k, (block + 1) * k)
        engine.state[:, columns].copy_(
            torch.randint(0, 2, (engine.n_max, k), device=engine.device,
                          dtype=engine.dtype, generator=generator))
    engine.best_states.copy_(engine.state)
    engine.best_energy.copy_(engine.problem.energy_func(engine.adjacency, engine.bias,
                                                        engine.state)[0])
    engine.improvements.zero_()


def decode_arms(engine, adj, bias, n, n_arms, *, count=4):
    """Best mask per arm block: A.3 on the ``count`` lowest-energy particles."""
    k = engine.num_k // n_arms
    sizes, masks = [], []
    for block in range(n_arms):
        start = block * k
        energy = engine.best_energy[start:start + k].reshape(-1)
        index = torch.argsort(energy, dim=0)[: int(count)] + start
        subset = engine.best_states[:n][:, index].contiguous()
        solutions = engine.problem.decode_gradient_flip(subset, adj, bias,
                                                       return_solutions=True)
        column_sizes = solutions.sum(dim=0)
        best = int(torch.argmax(column_sizes))
        picked = solutions[:, best].to(torch.uint8).cpu().numpy()
        masks.append(int.from_bytes(np.packbits(picked, bitorder="little").tobytes(), "little"))
        sizes.append(int(column_sizes[best]))
    return sizes, masks


def pack_mask(problem, state, adj, bias, n):
    """Single-mask decode of ``state`` (used for the pre-run archive box)."""
    solutions = problem.decode_gradient_flip(state[:n], adj, bias, return_solutions=True)
    sizes = solutions.sum(dim=0)
    best = int(torch.argmax(sizes))
    picked = solutions[:, best].to(torch.uint8).cpu().numpy()
    return int.from_bytes(np.packbits(picked, bitorder="little").tobytes(), "little")


# --------------------------------------------------------------------------- #
# mode 1: c(N)
# --------------------------------------------------------------------------- #
def timed_blocks(engine, blocks, chunk):
    """Median ms/step of the captured block, one ``sync`` per block (A0 semantics)."""
    engine.graph_exec.replay()
    sync(engine.device)
    times = []
    for _ in range(blocks):
        started = perf_counter()
        engine.graph_exec.replay()
        torch.cuda.synchronize(engine.device)
        times.append((perf_counter() - started) * 1000.0 / chunk)
    return times


def timed_loop(function, iters, device):
    """Median-free amortised ms per call: ``iters`` calls, one sync at the end.

    Syncing inside the loop measures *latency* (a sync costs ~0.4 ms on this
    card), which is not what a captured block pays: inside one replay the ten
    steps are enqueued back to back.  Amortising over a loop is the comparable
    number.
    """
    function()
    sync(device)
    started = perf_counter()
    for _ in range(int(iters)):
        function()
    sync(device)
    return (perf_counter() - started) * 1000.0 / int(iters)


def breakdown(engine, iters, device, adj, bias, n):
    """Per-component cost of one *step*, amortised (no per-call sync)."""
    state = engine.state
    k = engine.num_k
    dense_raw = torch.zeros((engine.n_max, engine.n_max), dtype=engine.dtype, device=device)
    dense_raw.index_put_((engine.slots, engine.col), engine.raw)

    def forward():
        engine._logits()

    def energy_csr():
        engine.problem.energy_func(engine.adjacency, engine.bias, state)

    def energy_dense():
        engine.problem.energy_func(dense_raw, engine.bias, state)

    def sample():
        probs = torch.div(engine._logits(), engine.temperature).sigmoid()
        noise = torch.rand(state.shape, device=device, dtype=engine.dtype,
                           generator=engine.generator)
        flipped = torch.where(noise < probs, 1 - state, state) * engine.keep
        state.copy_(flipped)

    def decode():
        decode_arms(engine, adj, bias, n, 1, count=4)

    return dict(forward=timed_loop(forward, iters, device),
                energy_csr=timed_loop(energy_csr, iters, device),
                energy_dense=timed_loop(energy_dense, iters, device),
                sample=timed_loop(sample, iters, device),
                decode=timed_loop(decode, max(5, iters // 10), device))


def bench(args):
    solver, _ranker, dataset = setup(args)
    device = solver.accelerator.device
    n_max, capacity = shape_of(dataset, args.padded_shapes)
    graph = dataset[0].to(device)
    edges = graph.edge_index
    n = int(graph.num_nodes)
    adj, bias = adjacency(graph), graph.b
    print(f"bench: graph_000000 nodes={n} edges={edges.shape[1]} "
          f"n_max={n_max} capacity={capacity} num_k_base={BASE_K}", flush=True)
    blocks = max(1, int(args.bench_steps) // int(args.chunk))
    rows = []
    for n_arms in args.bench_arms:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        engine = engine_for(solver, n_arms, n_max, capacity, args, device)
        engine.load(graph, edges, ARM_SEEDS[0])
        arms_from(engine, n_arms)
        times = timed_blocks(engine, blocks, args.chunk)
        step_ms = median(times)
        parts = breakdown(engine, 50, device, adj, bias, n)
        peak = torch.cuda.max_memory_allocated(device) / 2 ** 30
        step_parts = parts["forward"] + parts["energy_csr"] + parts["sample"]
        row = dict(n_arms=n_arms, num_k=engine.num_k, steps=blocks * int(args.chunk),
                   ms_step=round(step_ms, 4), ms_block=round(step_ms * int(args.chunk), 4),
                   p10=round(float(np.percentile(times, 10)), 4),
                   p90=round(float(np.percentile(times, 90)), 4),
                   peak_gib=round(peak, 3),
                   forward=round(parts["forward"], 4),
                   energy_csr=round(parts["energy_csr"], 4),
                   energy_dense=round(parts["energy_dense"], 4),
                   sample=round(parts["sample"], 4),
                   decode=round(parts["decode"], 4),
                   eager_sum=round(step_parts, 4),
                   captured_minus_eager=round(step_ms - step_parts, 4))
        rows.append(row)
        print(json.dumps(row), flush=True)
        del engine
    base = rows[0]["ms_step"]
    for row in rows:
        row["c"] = round(row["ms_step"] / base, 4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(protocol=dict(num_k_base=BASE_K, steps_bench=blocks * int(args.chunk),
                                 chunk=args.chunk, dense=bool(args.dense),
                                 dtype=str(solver.dtype), graph=f"graph_000000",
                                 nodes=n, n_max=n_max, capacity=capacity,
                                 seeds=list(ARM_SEEDS[:max(args.bench_arms)])),
                   rows=rows,
                   verdict=dict(n2_c=next(r["c"] for r in rows if r["n_arms"] == 2)
                                if any(r["n_arms"] == 2 for r in rows) else None,
                                n3_c=next(r["c"] for r in rows if r["n_arms"] == 3)
                                if any(r["n_arms"] == 3 for r in rows) else None,
                                n2_threshold=1.60, n3_threshold=1.99))
    (out / "c_N.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {out / 'c_N.json'}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# mode 2: 200-graph single vs parallel
# --------------------------------------------------------------------------- #
def _numeric(value):
    """CSV cells come back as strings; recover numbers for the summary maths."""
    text = str(value).strip()
    if not text:
        return value
    try:
        return int(text) if text.lstrip("-").isdigit() else float(text)
    except ValueError:
        return value


def compare(args):
    if getattr(args, "from_csv", None):
        with Path(args.from_csv).open(newline="", encoding="utf-8") as handle:
            rows = [{key: _numeric(value) for key, value in row.items()}
                    for row in csv.DictReader(handle)]
        plans = [(1, BASE_STEPS)]
        for n_arms, c in ((2, args.c2), (3, args.c3)):
            if c is None:
                continue
            plans.append((n_arms, int(math.floor(BASE_STEPS / float(c)) // args.chunk
                                      * args.chunk)))
        print(f"compare: reused {len(rows)} rows from {args.from_csv}, "
              f"plans={[(n, s) for n, s in plans]}", flush=True)
        return analyse(args, rows, plans)
    solver, ranker, dataset = setup(args)
    device = solver.accelerator.device
    n_max, capacity = shape_of(dataset, args.padded_shapes)
    graphs = min(int(args.graphs), len(dataset))
    c2 = float(args.c2 if args.c2 is not None else 1.0)
    c3 = float(args.c3 if args.c3 is not None else 1.0)
    plans = [(1, BASE_STEPS)]
    for n_arms, c in ((2, c2), (3, c3)):
        if args.c2 is None and n_arms == 2 or args.c3 is None and n_arms == 3:
            continue
        plans.append((n_arms, int(math.floor(BASE_STEPS / c) // args.chunk * args.chunk)))
    engines = {n_arms: engine_for(solver, n_arms, n_max, capacity, args, device)
               for n_arms, _ in plans}
    print(f"compare: {graphs} graphs, plans={[(n, s) for n, s in plans]}, "
          f"n_max={n_max} capacity={capacity}", flush=True)
    rows = []
    for position in range(graphs):
        graph = dataset[position].to(device)
        edges = graph.edge_index
        n = int(graph.num_nodes)
        adj, bias = adjacency(graph), graph.b
        topology = SearchGraph(n, edges, seed=1017)
        record = dict(graph_id=Path(dataset.files[position]).stem, nodes=n, edges=int(edges.shape[1]))
        for n_arms, steps in plans:
            engine = engines[n_arms]
            tick = perf_counter()
            engine.load(graph, edges, ARM_SEEDS[0])
            arms_from(engine, n_arms)
            blocks = max(1, steps // int(args.chunk))
            started = perf_counter()
            engine.run(float("inf"), lambda states: None, max_blocks=blocks,
                       decode_every=blocks + 1, sync_each_block=True)
            sync(device)
            search_seconds = perf_counter() - started
            sizes, masks = decode_arms(engine, adj, bias, n, n_arms, count=args.decode_count)
            tag = "single" if n_arms == 1 else f"par{n_arms}"
            record[f"{tag}_search"] = max(sizes)
            record[f"{tag}_arms"] = "/".join(str(size) for size in sizes)
            record[f"{tag}_steps"] = blocks * int(args.chunk)
            record[f"{tag}_ms"] = round(search_seconds * 1000.0 / (blocks * int(args.chunk)), 4)
            best_mask = masks[int(np.argmax(sizes))]
            if not is_independent(best_mask, topology.neighbors):
                raise ValueError(f"{record['graph_id']}: illegal set in {tag}")
            deadline = perf_counter() + min(args.post_reserve, 0.2 * 1.0)
            final = run_pave(topology, best_mask, ranker, args, deadline)
            sync(device)
            if not is_independent(final, topology.neighbors):
                raise ValueError(f"{record['graph_id']}: illegal PAVE set in {tag}")
            record[f"{tag}_final"] = final.bit_count()
            record[f"{tag}_wall"] = round(perf_counter() - tick, 4)
        rows.append(record)
        if position % 20 == 0 or position + 1 == graphs:
            print(f"  [{position + 1}/{graphs}] {record['graph_id']} "
                  f"single={record['single_search']}/{record['single_final']} "
                  + " ".join(f"{tag}={record[f'{tag}_search']}/{record[f'{tag}_final']}"
                             for tag, _ in ((f"par{n}", s) for n, s in plans)
                             if f"{tag}_search" in record),
                  flush=True)
    return analyse(args, rows, plans)


def analyse(args, rows, plans):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (out / "compare.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(graphs=len(rows), plans=[[n, s] for n, s in plans], delta={})
    for n_arms, steps in plans:
        if n_arms == 1:
            continue
        tag = f"par{n_arms}"
        for column in ("search", "final"):
            pairs = [(row[f"{tag}_{column}"], row[f"single_{column}"]) for row in rows]
            delta = [a - b for a, b in pairs]
            boot_mean, low, high = paired_bootstrap(delta, 5000, 1017)
            summary["delta"][f"{tag}_{column}"] = dict(
                mean=round(float(np.mean(delta)), 4), boot_mean=round(float(boot_mean), 4),
                ci=[round(low, 4), round(high, 4)],
                steps_per_arm=steps, wins=sum(1 for value in delta if value > 0),
                losses=sum(1 for value in delta if value < 0),
                ties=sum(1 for value in delta if value == 0))
    summary["mean_ms"] = {key: round(float(np.mean([row[key] for row in rows])), 4)
                          for key in fields if key.endswith("_ms")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)
    return 0


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = argv[0] if argv else "bench"
    args = parsed(argv[1:])
    return bench(args) if mode == "bench" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
