"""KaMIS (ReduMIS) anytime curve at the same wall-clock budgets as the deployment path.

The project has one classical reference point so far: KaMIS at a fixed 4.1 s/graph
mean landed on 20.06 over the 100 validation graphs while the deployed
B + frozen PAVE pipeline was at 19.51.  That number cannot answer the question a
reviewer asks first ("who wins at 0.5 s and 1.0 s?"), because KaMIS was never run
under a budget.  ``mis_baselines.kamis`` already forwards ``--time_limit``, so this
script only adds the sweep and the paired summary.

    python -u models/eval_kamis_anytime.py --split validation --test_samples 100 \
        --kamis tmp/kamis_build/redumis --budgets 0.1 0.5 1.0 4.0 --seeds 1017 1019 \
        --result_dir outputs/v8_macro/positioning

The paired reference is the frozen ``baseline`` arm of ``outputs/v7_tccr/budget``
(B search + frozen PAVE at 0.5 s and 1.0 s, 100 graphs, seeds 1017/1019/1021).
KaMIS is single-threaded CPU work while RLNN is one GPU, so the two numbers are
compared as wall-clock on the same machine, which is the usual convention.
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import torch

from eval_v2 import VALID, dataset_at
from mis_baselines import kamis
from v7_runs import digest, provenance

ROOT = Path(__file__).resolve().parents[1]
KAMIS_SEED_OFFSET = 1000


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="validation", choices=("validation", "test"))
    parser.add_argument("--test_path", type=Path)
    parser.add_argument("--test_samples", type=int, default=100)
    parser.add_argument("--kamis", type=Path, required=True)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.5, 1.0, 4.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1017, 1019])
    parser.add_argument("--reference_seed", type=int, default=0)
    parser.add_argument("--reference_dir", type=Path, default=ROOT / "outputs/v7_tccr/budget")
    parser.add_argument("--result_dir", type=Path, default=ROOT / "outputs/v8_macro/positioning")
    parser.add_argument("--bootstrap", type=int, default=5000)
    return parser


def neighbours_of(graph):
    neighbours = [set() for _ in range(graph.num_nodes)]
    for source, target in graph.edge_index.T.tolist():
        neighbours[source].add(target)
        neighbours[target].add(source)
    return neighbours


def is_independent(selected, neighbours):
    chosen = [node for node in range(len(neighbours)) if selected[node]]
    return all(other not in neighbours[node] for i, node in enumerate(chosen) for other in chosen[i + 1:])


def paired_bootstrap(deltas, rounds, seed=20260101):
    """Graph-level bootstrap CI of the mean paired difference."""
    generator = torch.Generator().manual_seed(seed)
    values = torch.tensor(deltas, dtype=torch.float64)
    index = torch.randint(0, values.numel(), (rounds, values.numel()), generator=generator)
    means = values[index].mean(dim=1)
    return float(torch.quantile(means, 0.025)), float(torch.quantile(means, 0.975))


def load_reference(path):
    """Mean size per (graph_id, budget) of the frozen baseline arm."""
    if not path.is_file():
        return {}
    pooled = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["arm"] != "baseline":
                continue
            pooled.setdefault((row["graph_id"], float(row["budget"])), []).append(int(row["final_size"]))
    return {key: mean(values) for key, values in pooled.items()}


def main():
    parser = build_parser()
    args = parser.parse_args()
    binary = args.kamis.expanduser().resolve()
    if not binary.is_file():
        parser.error(f"KaMIS binary not found: {binary}")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    output = args.result_dir / "kamis"
    output.mkdir(exist_ok=True)

    dataset = dataset_at(args.test_path or str(VALID), torch.float32, args.test_samples,
                         official=args.split == "test")
    reference = load_reference(args.reference_dir / f"budget_ref{args.reference_seed}.csv")
    rows = []
    started = perf_counter()
    for index, graph in enumerate(dataset):
        graph_id = f"idx_{index}" if args.split == "test" else Path(dataset.files[index]).stem
        neighbours = neighbours_of(graph)
        for budget in args.budgets:
            for seed in args.seeds:
                tick = perf_counter()
                selected = kamis(graph, binary, seed, budget)
                seconds = perf_counter() - tick
                size = int(selected.sum())
                if not is_independent(selected, neighbours):
                    raise ValueError(f"{graph_id}: KaMIS returned a non-independent set")
                rows.append(dict(graph_id=graph_id, budget=budget, seed=seed, size=size,
                                 wall_time_s=round(seconds, 4)))
        if (index + 1) % 20 == 0 or index + 1 == len(dataset):
            summary = " ".join(
                f"{budget}s:{mean(row['size'] for row in rows if row['budget'] == budget):.3f}"
                for budget in args.budgets)
            print(f"{index + 1}/{len(dataset)} graphs kamis {summary}", flush=True)

    path = output / f"kamis_ref{args.reference_seed}.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    per_budget = {}
    for budget in args.budgets:
        sizes = [row["size"] for row in rows if row["budget"] == budget]
        times = [row["wall_time_s"] for row in rows if row["budget"] == budget]
        entry = dict(kamis_mean=mean(sizes), kamis_median=median(sizes),
                     wall_mean_s=mean(times), wall_max_s=max(times), runs=len(sizes))
        by_graph = {}
        for row in rows:
            if row["budget"] == budget:
                by_graph.setdefault(row["graph_id"], []).append(row["size"])
        pairs, deltas = [], []
        for graph_id, values in sorted(by_graph.items()):
            key = (graph_id, budget)
            if key not in reference:
                continue
            pairs.append((graph_id, mean(values), reference[key]))
            deltas.append(reference[key] - mean(values))
        if pairs:
            low, high = paired_bootstrap(deltas, args.bootstrap)
            entry.update(reference=f"outputs/v7_tccr/budget/budget_ref{args.reference_seed}.csv",
                         paired_graphs=len(pairs),
                         ours_mean=mean(ours for _, _, ours in pairs),
                         delta_mean=mean(deltas), delta_ci=[low, high],
                         wins=sum(1 for delta in deltas if delta > 0),
                         ties=sum(1 for delta in deltas if delta == 0),
                         losses=sum(1 for delta in deltas if delta < 0))
        else:
            entry.update(reference=None, note="no paired baseline rows for this budget")
        per_budget[f"{budget:g}s"] = entry
        print(f"budget={budget:g}s kamis {entry['kamis_mean']:.4f} "
              f"({entry['wall_mean_s']:.3f}s/graph) "
              + (f"ours {entry['ours_mean']:.4f} delta {entry['delta_mean']:+.4f} "
                 f"[{entry['delta_ci'][0]:+.4f},{entry['delta_ci'][1]:+.4f}] "
                 f"{entry['wins']}/{entry['ties']}/{entry['losses']}" if pairs else "unpaired"),
              flush=True)

    metadata = dict(version="v8_positioning", status="completed", mode="kamis_anytime",
                    split=args.split, samples=len(dataset), budgets=args.budgets,
                    seeds=args.seeds, kamis=str(binary), kamis_sha256=digest(binary),
                    wall_time_s=round(perf_counter() - started, 2),
                    per_budget=per_budget, config=vars(args), **provenance(args.reference_seed))
    (output / f"summary_ref{args.reference_seed}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
