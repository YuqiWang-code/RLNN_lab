"""t7 (C): can a FIFO tabu walk escape the *deployment* solution?

C4 was closed on a Greedy-MIS start (``models/probe_tabu_walk.py``): the FIFO fires
(ER ~ 0.067) but 8 rounds never improve the solution.  This probe repeats the walk
from the **deployment solution** -- the mask A0 actually ships (810 fixed steps ->
A.3 top-4 decode -> PAVE 0.05 s) -- because the paper's "+0.004" is the gain PAVE
extracts *from that* state, and a reviewer will ask whether the plateau is an
artefact of the greedy start.

Two stages (``--stage all`` runs both):

``build``   GPU, A0 flags: 810 fixed steps -> top-4 decode -> PAVE (0.05 s) per graph;
            writes ``deploy.json`` plus ``edges.pt`` (topologies for the CPU stage).
``escape``  CPU: per graph, 8 outer rounds x 0.2 s from (a) the deployment mask and
            (b) the Greedy-MIS control, each at h=8 (FIFO on) and h=0 (empty FIFO =
            "the walk moves but forbids nothing").  Every returned set is asserted
            independent.

    python models/probe_deploy_escape.py --graphs 200 --stage all \
        --out outputs/v11/t7/escape
"""
import csv
import json
from math import inf
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from eval_v7_budget import run_pave
from eval_v8_padded import build_parser, shape_of
from mis_v1 import adjacency
from mis_v3_search import SearchGraph
from mis_v11_tabu import TabuList, platform_search_tabu
from mis_v12_parallel import decode_arms, engine_for, setup
from probe_tabu_walk import feasible, greedy_mis, summarise

STARTS = ("deploy", "greedy")


def escape_parser(argv=None):
    parser = build_parser()
    parser.add_argument("--graphs", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=8, help="outer rounds per walk")
    parser.add_argument("--escape_budget", type=float, default=0.2, help="seconds per walk")
    parser.add_argument("--deploy_steps", type=int, default=808, help="A0 fixed steps")
    parser.add_argument("--deploy_reserve", type=float, default=0.05, help="PAVE seconds")
    parser.add_argument("--h", type=int, default=8, help="FIFO depth (C4's window was 8..128)")
    parser.add_argument("--mode", default="cross", help="FIFO convention (L04 Alg.1)")
    parser.add_argument("--stage", choices=("all", "build", "escape"), default="all")
    parser.add_argument("--out", default="outputs/v11/t7/escape")
    parser.set_defaults(budgets=[1.0], seeds=[1017], repeats=1, shards=1, shard=0,
                        decoders=["captured"], num_k=200, chunk=10, dense=True,
                        mixed_precision="fp16", reference_seed=0,
                        result_dir=Path("outputs/v11/t7/escape_unused"),
                        save_dir=Path("outputs/v2/none_seed0/checkpoints"),
                        outer_rounds=8, exchange_rounds=3, horizon=2,
                        candidate_cap=32, improving_cap=48, repair_rounds=4)
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# stage 1: deployment solution (GPU)
# --------------------------------------------------------------------------- #
def build(args, out):
    solver, ranker, dataset = setup(args)
    device = solver.accelerator.device
    n_max, capacity = shape_of(dataset, args.padded_shapes)
    engine = engine_for(solver, 1, n_max, capacity, args, device)
    blocks = (args.deploy_steps + args.chunk - 1) // args.chunk
    seed = int(args.seeds[0])
    engine.load(dataset[0].to(device), dataset[0].edge_index, seed)
    engine.run(perf_counter() + 0.05, lambda states: None)
    torch.cuda.synchronize(device)

    rows, edges_kept, started = [], [], perf_counter()
    for index in range(len(dataset)):
        graph = dataset[index]
        edges = graph.edge_index
        n = int(graph.num_nodes)
        on_device = graph.to(device)
        adj = adjacency(on_device)
        engine.load(on_device, edges, seed)
        engine.run(inf, lambda states: None, max_blocks=blocks)
        # decode_arms slices best_states[:n]; the engine's bias is padded to n_max,
        # so hand it bias[:n] (real-node adjacency + real-node bias = A0's decode).
        sizes, masks = decode_arms(engine, adj, engine.bias[:n], n, 1, count=4)
        topology = SearchGraph(n, edges, seed=seed)
        deploy = run_pave(topology, masks[0], ranker, args,
                          perf_counter() + args.deploy_reserve)
        if not feasible(topology, deploy):
            raise SystemExit(f"PAVE returned a non-independent set on graph {index}")
        rows.append(dict(index=index, nodes=n, search_size=sizes[0],
                         deploy_size=deploy.bit_count(), deploy_mask=hex(deploy)))
        edges_kept.append(edges.cpu())
        if (index + 1) % 50 == 0:
            print(f"  build {index + 1}/{len(dataset)} "
                  f"deploy={mean(r['deploy_size'] for r in rows):.4f}", flush=True)
    torch.save(edges_kept, out / "edges.pt")
    payload = dict(graphs=len(rows), steps=blocks * args.chunk, seeds=[seed],
                   deploy_reserve=args.deploy_reserve,
                   wall_s=round(perf_counter() - started, 1), rows=rows)
    (out / "deploy.json").write_text(json.dumps(payload, indent=1))
    print(f"build: {len(rows)} graphs, mean search "
          f"{mean(r['search_size'] for r in rows):.4f} -> deploy "
          f"{mean(r['deploy_size'] for r in rows):.4f}, wall {payload['wall_s']} s",
          flush=True)


# --------------------------------------------------------------------------- #
# stage 2: walks (CPU)
# --------------------------------------------------------------------------- #
def walk(graph, start, *, h, mode, args):
    tabu = TabuList(h, mode=mode)
    records = []
    deadline = perf_counter() + args.escape_budget
    result = platform_search_tabu(
        graph, start, tabu=tabu, choose="coverage", outer_rounds=args.rounds,
        initial_rounds=3, horizon=args.horizon, candidate_cap=args.candidate_cap,
        improving_cap=args.improving_cap, repair_rounds=args.repair_rounds,
        deadline=deadline, on_round=records.append)
    return result, records, tabu


def escape(args, out, payload):
    edges_all = torch.load(out / "edges.pt", map_location="cpu", weights_only=False)
    depths = [0] + ([args.h] if args.h else [])
    rows = []
    started = perf_counter()
    for row in payload["rows"]:
        index, n, edges = row["index"], row["nodes"], edges_all[row["index"]]
        graph = SearchGraph(n, edges, seed=int(args.seeds[0]))
        starts = {"deploy": int(row["deploy_mask"], 16), "greedy": greedy_mis(graph)}
        for start_kind in STARTS:
            start = starts[start_kind]
            for h in depths:
                result, records, tabu = walk(graph, start, h=h, mode=args.mode, args=args)
                if not feasible(graph, result.best):
                    raise SystemExit(f"illegal set: graph {index}, start={start_kind}, h={h}")
                stats = summarise(records, args.rounds, start.bit_count(),
                                  result.best.bit_count())
                rows.append(dict(
                    graph_index=index, nodes=n, start=start_kind, h=h,
                    start_size=start.bit_count(), best_size=result.best.bit_count(),
                    gain=stats["gain"], er=stats["er"], er_pooled=stats["er_pooled"],
                    er_rounds=stats["er_rounds"], er_rejected=stats["er_rejected"],
                    ar=stats["ar"], refreshed_rounds=stats["refreshed_rounds"],
                    fifo_checks=tabu.checks, fifo_hits=tabu.hits,
                    rounds=len(records), deadline_hit=int(result.deadline_hit)))
        if (index + 1) % 50 == 0:
            print(f"  escape {index + 1}/{len(payload['rows'])}", flush=True)

    out_csv = out / "escape.csv"
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    per_cell = {}
    for start_kind in STARTS:
        for h in depths:
            subset = [r for r in rows if r["start"] == start_kind and r["h"] == h]
            gains = [r["gain"] for r in subset]
            per_cell[f"{start_kind}:h{h}"] = dict(
                graphs=len(subset),
                start_size=mean(r["start_size"] for r in subset),
                best_size=mean(r["best_size"] for r in subset),
                gain=mean(gains),
                gain_positive_graphs=sum(1 for g in gains if g > 0),
                best_gain=max(gains),
                refreshed_rounds=mean(r["refreshed_rounds"] for r in subset),
                refresh_graphs=sum(1 for r in subset if r["refreshed_rounds"] > 0),
                ar=mean(r["ar"] for r in subset),
                er=mean(r["er"] for r in subset if r["er"] == r["er"]) if any(
                    r["er"] == r["er"] for r in subset) else None,
                fifo_hits=sum(r["fifo_hits"] for r in subset),
                fifo_checks=sum(r["fifo_checks"] for r in subset),
                deadline_hits=sum(r["deadline_hit"] for r in subset))
    summary = dict(
        graphs=payload["graphs"], rounds=args.rounds, escape_budget=args.escape_budget,
        h=args.h, mode=args.mode, steps=payload["steps"],
        deploy_reserve=payload["deploy_reserve"],
        build_wall_s=payload["wall_s"], escape_wall_s=round(perf_counter() - started, 1),
        per_cell=per_cell)
    (out / "escape.json").write_text(json.dumps(summary, indent=1))
    for key, cell in per_cell.items():
        print(f"{key:12s} graphs={cell['graphs']:3d} start={cell['start_size']:.4f} "
              f"best={cell['best_size']:.4f} gain={cell['gain']:+.4f} "
              f"gain>0 {cell['gain_positive_graphs']:3d} graph(s) "
              f"refresh={cell['refreshed_rounds']:.3f} ({cell['refresh_graphs']} graph(s)) "
              f"AR={cell['ar']:.4f} ER={cell['er'] if cell['er'] is None else round(cell['er'], 4)} "
              f"hits={cell['fifo_hits']}/{cell['fifo_checks']} deadline={cell['deadline_hits']}",
              flush=True)


def main(argv=None):
    args = escape_parser(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage in ("all", "build"):
        build(args, out)
    if args.stage == "build":
        return
    payload = json.loads((out / "deploy.json").read_text())
    escape(args, out, payload)


if __name__ == "__main__":
    main()
