"""Cross-family regime probe: does the frozen RB-trained policy transfer?

The frozen backbone never saw Erdős–Rényi, Barabási–Albert, or graphs with more than
300 nodes.  This probe holds the clock fixed, gives KaMIS the same *algorithm* time as
a favour, and asks two questions:

1. does the +0.45 gap over KaMIS measured on RB 200–300 survive out of distribution;
2. does the PAVE repair on a weak constructor (measured +0.71 on RB) survive.

Three arms per graph, one search seed, one clock:

* ``rl``     —— deployment path: captured 10-step blocks, captured A.3 decode with the
                skip-unchanged cache, then frozen PAVE (horizon=1) inside the same clock;
* ``kamis``  —— ReduMIS with ``--time_limit`` equal to the full clock (the baseline gets
                its whole budget as *algorithm* time, so its wall clock is longer);
* ``greedy`` —— ``minimum_degree`` then PAVE with a 0.1 s clock; ``raw_size`` records the
                pre-PAVE size so the repair effect stays separable.

    python -u models/eval_regime.py --families RB_small_val ER_200_300 BA_200_300 RB_large \
        --graphs 40 --budgets 1.0 --reference_seed 0 --result_dir outputs/v8_macro/regime
"""
import csv
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from eval_pave_external import as_mask
from eval_v2 import dataset_at, load_policy, sync, write_run
from eval_v7_budget import POST_RESERVE, run_pave
from eval_v8_capture_budget import build_capture_parser, captured_search, paired_bootstrap
from mis_baselines import kamis, minimum_degree
from mis_v1 import adjacency
from mis_v3_search import SearchGraph
from mis_v3_value import PAVEV3
from mis_v7_recombine import is_independent
from mis_v8_decode import CapturedDecoder
from v7_runs import backbone, pave, provenance

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("rl", "kamis", "greedy")


def build_parser():
    parser = build_capture_parser()
    parser.add_argument("--families", nargs="+", default=["RB_small_val"],
                        help="subdirectories of --data_root, one per graph family")
    parser.add_argument("--data_root", type=Path, default=ROOT / "models/data/MIS_tensors")
    parser.add_argument("--graphs", type=int, default=40)
    parser.add_argument("--pave_limit", type=float, default=0.1,
                        help="PAVE clock for the greedy repair arm")
    parser.add_argument("--kamis", type=Path, default=ROOT / "tmp/kamis_build/redumis")
    parser.set_defaults(num_t=100, horizon=1, decoders=["cached"], arms=["captured"],
                        post_reserve=POST_RESERVE)
    return parser


def load_ranker(reference_seed):
    checkpoint = torch.load(pave(reference_seed), map_location="cpu", weights_only=False)
    ranker = PAVEV3(checkpoint["mode"], checkpoint.get("stats_dim", 24))
    ranker.load_state_dict(checkpoint["model_state"])
    return ranker.cpu().eval()


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.kamis.is_file():
        parser.error(f"missing KaMIS binary: {args.kamis}")
    # The frozen backbone is selected by reference seed, not by --test_path.
    args.save_dir = str(backbone(args.reference_seed).parent)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    output = args.result_dir / "regime"
    output.mkdir(exist_ok=True)

    solver = load_policy(args)
    solver.model.requires_grad_(False)
    ranker = load_ranker(args.reference_seed)
    device = solver.accelerator.device
    extended = 20000

    families = []
    for name in args.families:
        path = args.data_root / name
        if not (path / "tensor_manifest.json").is_file():
            parser.error(f"{path} has no tensor_manifest.json")
        dataset = dataset_at(str(path), solver.dtype, args.graphs, official=False)
        families.append((name, dataset))
    if not families:
        parser.error("no families to evaluate")

    # Warm the frozen kernels, the decode, PAVE and one capture outside every clock.
    warm = families[0][1][0]
    warm_edges = warm.edge_index
    warm = warm.to(device)
    warm_adj = adjacency(warm)
    warm_topology = SearchGraph(warm.num_nodes, warm_edges, seed=args.seeds[0])
    solver.num_t = extended
    warm_selected, _, _, _, _ = captured_search(solver, warm, warm_adj, args.seeds[0],
                                                perf_counter() + 0.2)
    run_pave(warm_topology, warm_selected, ranker, args, perf_counter() + 0.1)
    del warm, warm_adj, warm_topology
    sync(device)

    started = perf_counter()
    rows = []
    for name, dataset in families:
        for index, graph in enumerate(dataset):
            graph_id = Path(dataset.files[index]).stem
            edges = graph.edge_index
            nodes = graph.num_nodes
            graph = graph.to(device)
            adj = adjacency(graph)
            for seed in args.seeds:
                for budget in args.budgets:
                    topology = SearchGraph(nodes, edges, seed=seed)
                    neighbours = topology.neighbors
                    normalized = solver.policy_topology(graph, adj)
                    solver.num_t = extended

                    # rl: captured blocks + captured decode + frozen PAVE, all inside the clock.
                    sync(device)
                    tick = perf_counter()
                    deadline = tick + budget
                    cutoff = deadline - args.post_reserve
                    template = torch.zeros(nodes, solver.num_k, device=device, dtype=solver.dtype)
                    decoder = CapturedDecoder(solver.problem, adj, graph.b, template,
                                              chunk=args.decode_chunk)
                    selected, _, steps, decode_seconds, prelude = captured_search(
                        solver, graph, adj, seed, cutoff, normalized,
                        decoder.decode_best, skip_unchanged=True)
                    sync(device)
                    search_seconds = perf_counter() - tick
                    final = run_pave(topology, selected, ranker, args, deadline)
                    sync(device)
                    rl_seconds = perf_counter() - tick
                    if not is_independent(final, neighbours):
                        raise ValueError(f"{name}/{graph_id}: illegal rl set")
                    rows.append(dict(family=name, graph_id=graph_id, nodes=nodes,
                                     edges=edges.numel() // 2, search_seed=seed, budget=budget,
                                     arm="rl", size=final.bit_count(),
                                     raw_size=final.bit_count(),
                                     pave_gain=final.bit_count() - selected.bit_count(),
                                     steps=steps, wall_s=round(rl_seconds, 4),
                                     search_s=round(search_seconds, 4),
                                     decode_s=round(decode_seconds, 4),
                                     prelude_s=round(prelude, 4),
                                     ms_per_step=round(1e3 * search_seconds / max(steps, 1), 3),
                                     deadline_overrun=int(rl_seconds > budget)))

                    # kamis: the whole budget as algorithm time (favours the baseline).
                    tick = perf_counter()
                    solution = kamis(graph, args.kamis, seed, budget)
                    kamis_seconds = perf_counter() - tick
                    if not is_independent(as_mask(solution), neighbours):
                        raise ValueError(f"{name}/{graph_id}: illegal kamis set")
                    rows.append(dict(family=name, graph_id=graph_id, nodes=nodes,
                                     edges=edges.numel() // 2, search_seed=seed, budget=budget,
                                     arm="kamis", size=int(solution.sum()),
                                     raw_size=int(solution.sum()), pave_gain=0,
                                     steps=0, wall_s=round(kamis_seconds, 4),
                                     search_s=round(kamis_seconds, 4), decode_s=0.0,
                                     prelude_s=0.0, ms_per_step=0.0, deadline_overrun=0))

                    # greedy: weak constructor then PAVE, the +0.71 repair path on RB.
                    tick = perf_counter()
                    raw = minimum_degree(graph)
                    raw_mask = as_mask(raw)
                    paved = run_pave(topology, raw_mask, ranker, args, tick + args.pave_limit)
                    sync(device)
                    greedy_seconds = perf_counter() - tick
                    if not is_independent(raw_mask, neighbours):
                        raise ValueError(f"{name}/{graph_id}: illegal greedy set")
                    if not is_independent(paved, neighbours):
                        raise ValueError(f"{name}/{graph_id}: illegal greedy_pave set")
                    if paved.bit_count() < raw_mask.bit_count():
                        raise ValueError(f"{name}/{graph_id}: PAVE shrank a greedy set")
                    rows.append(dict(family=name, graph_id=graph_id, nodes=nodes,
                                     edges=edges.numel() // 2, search_seed=seed, budget=budget,
                                     arm="greedy", size=paved.bit_count(),
                                     raw_size=raw_mask.bit_count(),
                                     pave_gain=paved.bit_count() - raw_mask.bit_count(),
                                     steps=0, wall_s=round(greedy_seconds, 4),
                                     search_s=round(greedy_seconds, 4), decode_s=0.0,
                                     prelude_s=0.0, ms_per_step=0.0, deadline_overrun=0))
            if (index + 1) % 10 == 0 or index + 1 == len(dataset):
                parts = []
                for arm in ARMS:
                    chosen = [r for r in rows if r["family"] == name and r["arm"] == arm]
                    parts.append(f"{arm}:{mean(r['size'] for r in chosen):.3f}")
                print(f"{name} {index + 1}/{len(dataset)} " + " ".join(parts), flush=True)

    path = output / f"regime_ref{args.reference_seed}.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def paired(family, left, right, key="size"):
        picked = {}
        for graph_id in sorted({r["graph_id"] for r in rows if r["family"] == family}):
            for name in (left, right):
                values = [r for r in rows if r["family"] == family and r["graph_id"] == graph_id
                          and r["arm"] == name]
                if values:
                    picked.setdefault(name, {})[graph_id] = mean(v[key] for v in values)
        shared = sorted(set(picked[left]) & set(picked[right]))
        diffs = [picked[left][g] - picked[right][g] for g in shared]
        effect, low, high = paired_bootstrap(diffs, 5000, seed=20260101)
        return dict(n=len(diffs), mean=effect, low=low, high=high,
                    wins=sum(d > 0 for d in diffs), ties=sum(d == 0 for d in diffs),
                    losses=sum(d < 0 for d in diffs))

    def repair_stats(family):
        gains = []
        for graph_id in sorted({r["graph_id"] for r in rows if r["family"] == family}):
            values = [r["pave_gain"] for r in rows
                      if r["family"] == family and r["graph_id"] == graph_id and r["arm"] == "greedy"]
            if values:
                gains.append(mean(values))
        effect, low, high = paired_bootstrap(gains, 5000, seed=20260101)
        return dict(n=len(gains), mean=effect, low=low, high=high,
                    wins=sum(g > 0 for g in gains), ties=sum(g == 0 for g in gains),
                    losses=sum(g < 0 for g in gains))

    summary = {}
    for name, _ in families:
        chosen = [r for r in rows if r["family"] == name]
        summary[name] = dict(
            graphs=len({r["graph_id"] for r in chosen}),
            nodes=mean(r["nodes"] for r in chosen if r["arm"] == "rl"),
            edges=mean(r["edges"] for r in chosen if r["arm"] == "rl"),
            arms={arm: dict(size=mean(r["size"] for r in chosen if r["arm"] == arm),
                            wall_s=mean(r["wall_s"] for r in chosen if r["arm"] == arm))
                  for arm in ARMS},
            greedy_raw=mean(r["raw_size"] for r in chosen if r["arm"] == "greedy"),
            steps=mean(r["steps"] for r in chosen if r["arm"] == "rl"),
            paired=dict(rl_gt_kamis=paired(name, "rl", "kamis"),
                        rl_gt_greedy=paired(name, "rl", "greedy"),
                        greedy_repair=repair_stats(name)))

    metadata = dict(ref=args.reference_seed, seeds=args.seeds, budgets=args.budgets,
                    families=list(args.families), graphs=args.graphs, arms=list(ARMS),
                    horizon=args.horizon, post_reserve=args.post_reserve,
                    decode_chunk=args.decode_chunk, pave_limit=args.pave_limit,
                    kamis=str(args.kamis), wall_time_s=round(perf_counter() - started, 2),
                    summary=summary, config={k: str(v) for k, v in vars(args).items()},
                    **provenance(args.reference_seed))
    (output / f"summary_ref{args.reference_seed}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    write_run(output, metadata)

    print(f"{'family':<16}{'n':>6}{'rl':>9}{'kamis':>9}{'greedy':>9}"
          f"{'rl-kamis':>22}{'greedy repair':>22}", flush=True)
    for name in args.families:
        cell = summary[name]
        rk = cell["paired"]["rl_gt_kamis"]
        gr = cell["paired"]["greedy_repair"]
        print(f"{name:<16}{cell['nodes']:>6.0f}{cell['arms']['rl']['size']:>9.3f}"
              f"{cell['arms']['kamis']['size']:>9.3f}{cell['arms']['greedy']['size']:>9.3f}"
              f"{rk['mean']:>+9.3f} [{rk['low']:+.3f},{rk['high']:+.3f}]"
              f"{gr['mean']:>+9.3f} [{gr['low']:+.3f},{gr['high']:+.3f}]", flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
