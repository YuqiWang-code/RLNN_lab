"""t7 (B): the P1 closure law on 1000 graphs (input-quality ladder).

P1 (direction doc §14.2) says the gain PAVE extracts from an input solution is a
linear function of that input's **density** rho = |S| / n:

    C = -131.5 * rho + 11.84        (36 graphs, rung means, R^2 = 0.989)
    rho* = 0.0901                   (linear extrapolation of C = 0)
    C == 0 at the 1 s input         (the "strict upper bound = 0" claim)

That fit was scoped to 36 graphs and rung means; this probe recomputes it on the
full 1000-graph test set with the deployment ladder:

    greedy (min-degree, CPU) | 0.1 s | 0.2 s | 0.5 s | 1.0 s (A0, fixed 808 steps)

Every search rung comes from an existing artifact (read-only) and its C is the
``pave_gain`` column the eval driver already computed with the **deployment**
reserve ``min(0.05, 0.2 * budget)``; the greedy rung is measured here with a fixed
0.05 s PAVE so its C is comparable to the 1 s rung.  Rung means are fitted, not
per-graph points (P1's R^2 = 0.989 is a rung-mean number; the per-point R^2 = 0.27
must be quoted next to it).

    python models/probe_p1_law_1000.py --graphs 1000 --out outputs/v11
"""
import argparse
import csv
import json
from collections import defaultdict
from math import inf
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from eval_v7_budget import run_pave
from eval_v8_padded import build_parser
from mis_v3_search import SearchGraph
from mis_v3_value import PAVEV3
from probe_tabu_walk import greedy_mis
from v7_runs import pave

RUNGS = ("greedy", "0.1", "0.2", "0.5", "1.0")


def law_parser(argv=None):
    parser = build_parser()
    parser.add_argument("--graphs", type=int, default=1000)
    parser.add_argument("--pave_budget", type=float, default=0.05,
                        help="PAVE seconds for the greedy rung")
    parser.add_argument("--rung_1s",
                        default="outputs/v11/gate0/a0_steps808_n1000/padded/padded_ref0.csv")
    parser.add_argument("--rung_02",
                        default="outputs/v11/t7/a02_s1017_1000/padded/padded_ref0.csv")
    parser.add_argument("--rung_ladder",
                        default="outputs/v11/t7/ladder_s1017_1000/padded/padded_ref0.csv")
    parser.add_argument("--out", default="outputs/v11")
    parser.set_defaults(budgets=[1.0], seeds=[1017], repeats=1, shards=1, shard=0,
                        decoders=["captured"], num_k=200, dense=True,
                        mixed_precision="fp16", reference_seed=0,
                        result_dir=Path("outputs/v11/p1_law_unused"),
                        save_dir=Path("outputs/v2/none_seed0/checkpoints"),
                        outer_rounds=8, exchange_rounds=3, horizon=2,
                        candidate_cap=32, improving_cap=48, repair_rounds=4)
    return parser.parse_args(argv)


def read_rung(path):
    """budget -> {graph_id: (search_size, final_size, pave_gain)} from an eval CSV."""
    table = defaultdict(dict)
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            budget = float(row["budget"])
            table[budget][row["graph_id"]] = (
                float(row["search_size"]), float(row["final_size"]),
                float(row.get("pave_gain") or 0.0))
    return table


def spearman(left, right):
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        for position, index in enumerate(order):
            out[index] = position + 1.0
        return out
    if len(left) < 3:
        return float("nan")
    a, b = ranks(left), ranks(right)
    ma, mb = mean(a), mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else float("nan")


def main(argv=None):
    args = law_parser(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    graphs = []
    solver = None
    dataset = None
    from eval_v2 import dataset_at  # local import: keeps --help offline

    dataset = dataset_at(args.test_path, torch.float32, args.graphs)
    checkpoint = torch.load(pave(args.reference_seed), map_location="cpu", weights_only=False)
    ranker = PAVEV3(checkpoint["mode"], checkpoint.get("stats_dim", 24))
    ranker.load_state_dict(checkpoint["model_state"])
    ranker = ranker.cpu().eval()
    seed = int(args.seeds[0])

    per_rung = {rung: {} for rung in RUNGS}       # rung -> graph_id -> (C, rho)
    # ---------------------------------------------------------------- greedy rung
    started = perf_counter()
    for index in range(len(dataset)):
        graph_obj = dataset[index]
        n = int(graph_obj.num_nodes)
        edges = graph_obj.edge_index
        graph = SearchGraph(n, edges, seed=seed)
        start = greedy_mis(graph)
        best = run_pave(graph, start, ranker, args, perf_counter() + args.pave_budget)
        size_before = start.bit_count()
        graphs.append(dict(graph_id=Path(dataset.files[index]).stem, nodes=n))
        per_rung["greedy"][graphs[-1]["graph_id"]] = (
            float(best.bit_count() - size_before), size_before / n)
        if (index + 1) % 200 == 0:
            print(f"  greedy {index + 1}/{len(dataset)} ({perf_counter() - started:.0f} s)",
                  flush=True)

    # -------------------------------------------------------------- search rungs
    sources = {"1.0": (args.rung_1s, 1.0), "0.2": (args.rung_02, 0.2)}
    ladder = read_rung(args.rung_ladder)
    for budget in ("0.1", "0.5"):
        sources[budget] = (args.rung_ladder, float(budget))
    for rung, (path, budget) in sources.items():
        table = read_rung(path)[budget]
        missing = [g["graph_id"] for g in graphs if g["graph_id"] not in table]
        if missing:
            raise SystemExit(f"rung {rung}: {len(missing)} graphs missing from {path}")
        for g in graphs:
            search_size, final_size, gain = table[g["graph_id"]]
            per_rung[rung][g["graph_id"]] = (final_size - search_size, search_size / g["nodes"])
        print(f"  rung {rung}: {len(table)} rows from {path}", flush=True)

    # ----------------------------------------------------------------- statistics
    rung_stats, points = {}, []
    for rung in RUNGS:
        cells = per_rung[rung]
        cs = [cells[g["graph_id"]][0] for g in graphs]
        rhos = [cells[g["graph_id"]][1] for g in graphs]
        rung_stats[rung] = dict(
            graphs=len(cs), mean_c=mean(cs), mean_rho=mean(rhos),
            c_positive_graphs=sum(1 for c in cs if c > 0),
            c_positive_fraction=sum(1 for c in cs if c > 0) / len(cs),
            c_max=max(cs), c_min=min(cs))
        points.append((mean(rhos), mean(cs), rung))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else float("nan")
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2_rung = 1 - ss_res / ss_tot if ss_tot else float("nan")
    zero = -intercept / slope if slope else float("nan")

    per_graph = [(per_rung[r][g["graph_id"]][0], per_rung[r][g["graph_id"]][1])
                 for g in graphs for r in RUNGS]
    gx = [p[1] for p in per_graph]
    gy = [p[0] for p in per_graph]
    mx2, my2 = mean(gx), mean(gy)
    sxx2 = sum((x - mx2) ** 2 for x in gx)
    sxy2 = sum((x - mx2) * (y - my2) for x, y in zip(gx, gy))
    slope2 = sxy2 / sxx2 if sxx2 else float("nan")
    inter2 = my2 - slope2 * mx2
    ss_tot2 = sum((y - my2) ** 2 for y in gy)
    ss_res2 = sum((y - (slope2 * x + inter2)) ** 2 for x, y in zip(gx, gy))
    r2_point = 1 - ss_res2 / ss_tot2 if ss_tot2 else float("nan")

    strict = rung_stats["1.0"]["c_positive_fraction"] <= 0.02
    summary = dict(
        graphs=len(graphs), rungs=RUNGS, pave_budget_greedy=args.pave_budget,
        pave_reserve_note="search rungs use the deployment reserve min(0.05, 0.2*budget)",
        sources=dict(greedy="computed here", **{r: s[0] for r, s in sources.items()}),
        per_rung=rung_stats,
        fit_rung_means=dict(slope=slope, intercept=intercept, r2=r2_rung, zero=zero,
                            points=[dict(rung=p[2], rho=p[0], c=p[1]) for p in points]),
        fit_per_point=dict(slope=slope2, intercept=inter2, r2=r2_point),
        spearman_c_rho=spearman(gy, gx),
        one_second_bound="strict" if strict else "empirical",
        one_second_c_positive_fraction=rung_stats["1.0"]["c_positive_fraction"],
        wall_s=round(perf_counter() - started, 1))
    (out / "p1_law_1000.json").write_text(json.dumps(summary, indent=1))

    rows = []
    for g in graphs:
        for rung in RUNGS:
            c, rho = per_rung[rung][g["graph_id"]]
            rows.append(dict(graph_id=g["graph_id"], rung=rung, nodes=g["nodes"],
                             rho=rho, c=c))
    with (out / "p1_law_1000.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["graph_id", "rung", "nodes", "rho", "c"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    for rung in RUNGS:
        s = rung_stats[rung]
        print(f"{rung:7s} graphs={s['graphs']} rho={s['mean_rho']:.5f} "
              f"C={s['mean_c']:+.4f} C>0 {s['c_positive_graphs']:4d} "
              f"({s['c_positive_fraction']:.4%}) max C={s['c_max']:.1f}", flush=True)
    print(f"fit (rung means): C = {slope:.3f}*rho + {intercept:.3f}  R2={r2_rung:.4f} "
          f"zero={zero:.5f}  spearman={spearman(gy, gx):.4f}", flush=True)
    print(f"fit (per point): C = {slope2:.3f}*rho + {inter2:.3f}  R2={r2_point:.4f}", flush=True)
    print(f"1 s bound: {summary['one_second_bound']} "
          f"(C>0 on {summary['one_second_c_positive_fraction']:.4%} of graphs)", flush=True)


if __name__ == "__main__":
    main()
