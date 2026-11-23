#!/usr/bin/env python
"""Export the smallest possible slice of the real RB test set for the MATLAB probes.

The MATLAB scripts in this directory must run with base MATLAB only, so every
number they cannot compute themselves (frozen policy solutions, PAVE solutions,
closure gaps, policy trajectories) is exported here as plain text.

What is exported (``--out``, default ``data/``):

``graphs.csv``          graph id, n, m, edge file name
``edges/g######.csv``   one undirected edge per line, ``u,v``, 1-based
``solutions.csv``       graph, kind, level, size, node list (1-based, space separated)
``ladder.csv``          per graph per input level: steps, search size, PAVE(0.1 s)
                        size/gain, unbudgeted platform closure size/gap, density
``trace_best.csv``      best-so-far size every ``--decode_every`` blocks, K=200 and K=20
``trace_pevents.csv``   per-particle size improvements (block, steps, size)
``trace_pcensor.csv``   per-particle observation window (censoring) and final size
``protocol_1000.csv``   the real 1000-graph per-graph sizes of the deployed 1 s table
                        (copied from ``outputs/v8_macro/author_test``), for the power probe

Everything else (destroy/rebuild, exact k-move search, two-phase budget DP,
survival fits) is done in MATLAB on these files.

Run on the training host (CUDA required)::

    cd /home/hzeng/project/ZH/RLNN_lab
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
        /home/hzeng/envs/rlnn/bin/python train_scripts/v11/validation/export_validation_data.py \
        --graphs 36 --out train_scripts/v11/validation/data
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "models"))

from eval_v2 import dataset_at, load_policy, sync                      # noqa: E402
from eval_v7_budget import run_pave                                   # noqa: E402
from eval_v8_padded import build_parser, decode_top_best              # noqa: E402
from mis_v1 import adjacency                                          # noqa: E402
from mis_baselines import minimum_degree                              # noqa: E402
from mis_v3_search import SearchGraph                                 # noqa: E402
from mis_v3_value import PAVEV3                                       # noqa: E402
from mis_v8_pad import PaddedSearch                                   # noqa: E402
from v7_runs import pave                                              # noqa: E402

TEST_SET = "models/data/MIS_tensors/RB_small_author_test"
PROTOCOL_CSV = "outputs/v8_macro/author_test/capture_budget/capture_budget_ref0.csv"


def eval_args(extra):
    """The deployed v9 configuration, parsed by the very same parser as the eval."""
    parser = build_parser()
    argv = [
        "--test_path", TEST_SET, "--test_samples", "1000", "--split", "test",
        "--save_dir", "outputs/v2/none_seed0/checkpoints", "--num_k", "200",
        "--reference_seed", "0", "--seeds", "1017", "--dense",
        "--decoders", "captured", "--budgets", "1.0",
        "--outer_rounds", "8", "--exchange_rounds", "3", "--horizon", "1",
        "--candidate_cap", "32", "--improving_cap", "48", "--repair_rounds", "4",
        "--result_dir", "/tmp/v11_export_unused",
    ]
    return parser.parse_args(argv + list(extra))


def node_list(mask, n):
    return [i + 1 for i in range(n) if (mask >> i) & 1]


def column_sizes(problem, states, adj, bias, cols, nodes):
    subset = states[: int(nodes), cols].contiguous()
    solutions = problem.decode_gradient_flip(subset, adj, bias, return_solutions=True)
    return [int(v) for v in solutions.sum(dim=0).cpu().numpy().tolist()]


def pick_indices(dataset, count, seed=7):
    """``count`` graphs spread over the n-quantiles of the split (different sizes)."""
    sizes = []
    for index in range(len(dataset)):
        graph = torch.load(dataset.files[index], map_location="cpu", weights_only=False)
        sizes.append(int(graph["num_nodes"]))
    order = sorted(range(len(dataset)), key=lambda i: (sizes[i], i))
    picks = sorted({order[int(round((k + 0.5) / count * (len(order) - 1)))]
                    for k in range(count)})
    return picks, sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=int, default=36)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--pave_budget", type=float, default=0.1,
                        help="PAVE clock used for the G-vs-C ladder (paper: 0.1 s)")
    parser.add_argument("--post_reserve", type=float, default=0.05)
    parser.add_argument("--reserve_frac", type=float, default=0.2)
    parser.add_argument("--trace_blocks_k200", type=int, default=140)
    parser.add_argument("--trace_blocks_k20", type=int, default=300)
    parser.add_argument("--decode_every", type=int, default=5)
    parser.add_argument("--particles", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1017)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "data"))
    parser.add_argument("--skip_protocol", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    (out / "edges").mkdir(parents=True, exist_ok=True)
    ev = eval_args([])
    ev.seeds = [args.seed]

    solver = load_policy(ev)
    solver.model.requires_grad_(False)
    solver.num_t = 20000  # the A.3 decode fixed point, as in eval_v8_padded
    checkpoint = torch.load(pave(ev.reference_seed), map_location="cpu", weights_only=False)
    ranker = PAVEV3(checkpoint["mode"], checkpoint.get("stats_dim", 24))
    ranker.load_state_dict(checkpoint["model_state"])
    ranker = ranker.cpu().eval()
    device = solver.accelerator.device
    dataset = dataset_at(ev.test_path, solver.dtype, 1000, official=True)
    picks, sizes = pick_indices(dataset, args.graphs)
    n_max, capacity = max(sizes), 105
    print(f"picked {len(picks)} graphs, n in [{min(sizes)} , {max(sizes)}]", flush=True)

    engine = PaddedSearch(solver, n_max=n_max, capacity=capacity, chunk=10,
                          device=device, decode_chunk=4, dense=True)
    engine.load(dataset[0].to(device), dataset[0].edge_index, args.seed)
    engine.run(perf_counter() + 0.05, lambda states: None)
    sync(device)
    solver.num_k = 20
    engine20 = PaddedSearch(solver, n_max=n_max, capacity=capacity, chunk=10,
                            device=device, decode_chunk=4, dense=True)
    engine20.load(dataset[0].to(device), dataset[0].edge_index, args.seed)
    engine20.run(perf_counter() + 0.05, lambda states: None)
    sync(device)
    solver.num_k = 200

    meta_rows, sol_rows, ladder_rows = [], [], []
    best_rows, event_rows, censor_rows = [], [], []

    for position, index in enumerate(picks):
        graph_cpu = dataset[index]
        graph = graph_cpu.to(device)
        n = int(graph.num_nodes)
        edges = graph_cpu.edge_index
        adj = adjacency(graph)
        bias = graph.b
        topology = SearchGraph(n, edges, seed=args.seed)
        neighbours = topology.neighbors
        name = Path(dataset.files[index]).stem
        uid = index

        source = edges.cpu().numpy()
        pairs = sorted({(int(min(a, b)), int(max(a, b)))
                        for a, b in zip(source[0].tolist(), source[1].tolist()) if a != b})
        edge_file = f"edges/e{uid:06d}.csv"
        with (out / edge_file).open("w", newline="") as handle:
            writer = csv.writer(handle)
            for u, v in pairs:
                writer.writerow([u + 1, v + 1])
        meta_rows.append(dict(id=uid, name=name, n=n, m=len(pairs), edges=edge_file))

        levels = {}
        for budget in args.budgets:
            tick = perf_counter()
            reserve = min(args.post_reserve, args.reserve_frac * budget)
            cutoff = tick + budget - reserve
            box = []

            def decode(best, box=box):
                candidate = decode_top_best(solver.problem, best, engine.best_energy, adj,
                                            bias, n, 4)
                box.append(candidate)

            engine.load(graph, edges, args.seed)
            decode(engine.best_states)
            steps, _, blocks = engine.run(cutoff, decode, decode_every=1000)
            sync(device)
            seconds = perf_counter() - tick
            selected = max(box, key=int.bit_count)
            levels[f"{budget:g}"] = dict(size=selected.bit_count(), steps=steps,
                                         seconds=seconds, mask=selected, budget=budget,
                                         reserve=reserve)
            sol_rows.append(dict(id=uid, kind="search", level=f"{budget:g}",
                                 size=selected.bit_count(),
                                 nodes=" ".join(map(str, node_list(selected, n)))))

        # The bottom of the quality ladder: the same min-degree greedy that
        # ``eval_pave_external --source min_degree`` uses (``mis_baselines.py:12``).
        greedy = minimum_degree(graph_cpu).to(torch.uint8).cpu().numpy()
        greedy_mask = 0
        for node in range(n):
            if greedy[node]:
                greedy_mask |= 1 << node
        levels["greedy"] = dict(size=greedy_mask.bit_count(), steps=0, seconds=0.0,
                                mask=greedy_mask, budget=0.0, reserve=0.0)
        sol_rows.append(dict(id=uid, kind="search", level="greedy",
                             size=greedy_mask.bit_count(),
                             nodes=" ".join(map(str, node_list(greedy_mask, n)))))

        for level, info in levels.items():
            selected = info["mask"]
            mark = perf_counter()
            pave_small = run_pave(topology, selected, ranker, ev, mark + args.pave_budget)
            sync(device)
            closed = run_pave(topology, selected, ranker, ev, mark + 3600.0)
            sync(device)
            gain = pave_small.bit_count() - selected.bit_count()
            gap = closed.bit_count() - selected.bit_count()
            ladder_rows.append(dict(
                id=uid, n=n, m=len(pairs), level=level, budget=info["budget"],
                steps=info["steps"], seconds=round(info["seconds"], 4),
                search_size=selected.bit_count(), rho=round(selected.bit_count() / n, 6),
                pave01_size=pave_small.bit_count(), pave01_gain=gain,
                c_inf_size=closed.bit_count(), c_inf_gap=gap))
            sol_rows.append(dict(id=uid, kind="pave01", level=level,
                                 size=pave_small.bit_count(),
                                 nodes=" ".join(map(str, node_list(pave_small, n)))))
            sol_rows.append(dict(id=uid, kind="c_inf", level=level,
                                 size=closed.bit_count(),
                                 nodes=" ".join(map(str, node_list(closed, n)))))
            if level == "1":
                mark2 = perf_counter()
                deployed = run_pave(topology, selected, ranker, ev, mark2 + args.post_reserve)
                sync(device)
                sol_rows.append(dict(id=uid, kind="pave_deployed", level="1.0",
                                     size=deployed.bit_count(),
                                     nodes=" ".join(map(str, node_list(deployed, n)))))
            print(f"  [{position + 1}/{len(picks)}] {name} n={n} m={len(pairs)} "
                  f"level={level} steps={info['steps']} size={selected.bit_count()} "
                  f"gain={gain:+d} gap={gap:+d}", flush=True)

        for tag, eng, blocks in (("K200", engine, args.trace_blocks_k200),
                                 ("K20", engine20, args.trace_blocks_k20)):
            columns = torch.arange(min(args.particles, eng.num_k), device=device)
            events = []
            records = []
            start = perf_counter()
            last = {}

            def on_block(count, eng=eng, columns=columns, start=start, events=events,
                         records=records, last=last):
                if count % args.decode_every:
                    return
                sizes_now = column_sizes(solver.problem, eng.best_states, adj, bias,
                                         columns, n)
                now = perf_counter() - start
                records.append((count, count * eng.chunk, round(now, 4),
                                max(sizes_now)))
                for column, value in enumerate(sizes_now):
                    if value > last.get(column, -1):
                        last[column] = value
                        events.append((column, count, count * eng.chunk, value))

            eng.load(graph, edges, args.seed)
            eng.run(start + 60.0, lambda states: None, max_blocks=blocks,
                    decode_every=10 ** 7, on_block=on_block)
            sync(device)
            whole = perf_counter() - start
            for count, steps, now, best in records:
                best_rows.append(dict(id=uid, K=tag, rec=count, steps=steps, t=now,
                                      best=best))
            for column, count, steps, value in events:
                event_rows.append(dict(id=uid, K=tag, particle=column + 1, block=count,
                                       steps=steps, size=value))
            for column in range(len(columns)):
                final = last.get(column, 0)
                censor_rows.append(dict(id=uid, K=tag, particle=column + 1,
                                        last_block=blocks, last_steps=blocks * eng.chunk,
                                        last_t=round(whole, 4), final_size=final))
            print(f"  [{position + 1}/{len(picks)}] {name} trace {tag} "
                  f"{blocks} blocks in {whole:.2f}s records={len(records)} "
                  f"events={len(events)}", flush=True)

        if (position + 1) % 5 == 0:
            write_csv(out, meta_rows, sol_rows, ladder_rows, best_rows, event_rows,
                      censor_rows)

    write_csv(out, meta_rows, sol_rows, ladder_rows, best_rows, event_rows, censor_rows)

    if not args.skip_protocol:
        source = Path(ROOT / PROTOCOL_CSV)
        sizes_by_id = {Path(dataset.files[i]).stem: sizes[i] for i in range(len(dataset))}
        rows = 0
        with source.open() as handle, (out / "protocol_1000.csv").open("w", newline="") as sink:
            reader = csv.DictReader(handle)
            writer = csv.writer(sink)
            writer.writerow(["graph_id", "n", "budget", "cell", "seed", "final_size"])
            for row in reader:
                if abs(float(row["budget"]) - 1.0) > 1e-9:
                    continue
                stem = f"graph_{int(row['graph_id'].split('_')[-1]):06d}"
                writer.writerow([row["graph_id"].split("_")[-1], sizes_by_id.get(stem, 0),
                                 row["budget"], row["cell"], row["search_seed"],
                                 row["final_size"]])
                rows += 1
        print(f"protocol_1000.csv: {rows} rows", flush=True)

    summary = dict(graphs=len(picks), budgets=args.budgets, seed=args.seed,
                   n_max=n_max, capacity=capacity,
                   pave_budget=args.pave_budget, post_reserve=args.post_reserve,
                   trace_blocks=dict(K200=args.trace_blocks_k200,
                                     K20=args.trace_blocks_k20),
                   decode_every=args.decode_every, particles=args.particles)
    (out / "export_meta.json").write_text(json.dumps(summary, indent=2))
    print("done", flush=True)


def write_csv(out, meta_rows, sol_rows, ladder_rows, best_rows, event_rows, censor_rows):
    def dump(name, rows):
        if not rows:
            return
        with (out / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    dump("graphs.csv", meta_rows)
    dump("solutions.csv", sol_rows)
    dump("ladder.csv", ladder_rows)
    dump("trace_best.csv", best_rows)
    dump("trace_pevents.csv", event_rows)
    dump("trace_pcensor.csv", censor_rows)


if __name__ == "__main__":
    main()
