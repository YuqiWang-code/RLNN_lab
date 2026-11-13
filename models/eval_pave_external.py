"""Feed an external solver's solution into the frozen PAVE post-processing.

The question this answers is the one a reviewer asks about PAVE: is it a
property of our own (weak, under-converged) search, or does it improve *any*
solution?  KaMIS/ReduMIS at a short ``--time_limit`` is the cleanest external
source: it is a strong, independent, CPU-only MIS solver, and its output is a
maximal independent set rather than something our policy touched.

Arms per graph (all start from the same KaMIS solution):

    kamis              the raw KaMIS output at ``--kamis_limit``
    kamis_extend       one greedy extension of that output (microseconds)
    kamis_pave         frozen PAVE, learned ranker, budget ``--pave_limit``
    kamis_pave_rule    frozen PAVE, the built-in ``coverage`` rule instead of the ranker
    kamis_pave_random  frozen PAVE, uniformly random move choice among the same pool
    kamis_long         KaMIS alone at ``kamis_limit + pave_limit`` (equal total clock)

``kamis_extend``/``kamis_pave_rule``/``kamis_pave_random`` are the ablations that
decide whether the learned ranker matters or whether "any local search helps".

    python -u models/eval_pave_external.py --kamis tmp/kamis_build/redumis \
        --kamis_limit 0.1 --pave_limit 0.1 --result_dir outputs/v8_macro/pave_external

The PAVE hyperparameters mirror the frozen deployment path (``eval_v3`` defaults:
outer_rounds 8, exchange_rounds 3, horizon 2, candidate_cap 32, improving_cap 48,
repair_rounds 4, anchored False) so the comparison is against the PAVE that the
paper uses, not a retuned one.
"""
import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from eval_v2 import VALID, dataset_at
from mis_baselines import kamis, minimum_degree
from mis_v3_search import SearchGraph, platform_search
from mis_v3_value import STATS_DIM, PAVEV3
from v7_runs import digest, pave, provenance

ROOT = Path(__file__).resolve().parents[1]
PAVE_ARGS = dict(outer_rounds=8, initial_rounds=3, horizon=2, candidate_cap=32,
                 improving_cap=48, repair_rounds=4, anchored=False)
ARMS = ("kamis", "kamis_extend", "kamis_pave", "kamis_pave_rule", "kamis_pave_random", "kamis_long")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="validation", choices=("validation", "test"))
    parser.add_argument("--test_path", type=Path)
    parser.add_argument("--test_samples", type=int, default=100)
    parser.add_argument("--kamis", type=Path, required=True)
    parser.add_argument("--source", default="kamis", choices=("kamis", "min_degree"))
    parser.add_argument("--kamis_limit", type=float, default=0.1)
    parser.add_argument("--pave_limit", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1017])
    parser.add_argument("--reference_seed", type=int, default=0)
    parser.add_argument("--result_dir", type=Path, default=ROOT / "outputs/v8_macro/pave_external")
    parser.add_argument("--bootstrap", type=int, default=5000)
    return parser


def neighbours_of(graph):
    neighbours = [set() for _ in range(graph.num_nodes)]
    for source, target in graph.edge_index.T.tolist():
        neighbours[source].add(target)
        neighbours[target].add(source)
    return neighbours


def as_bool(selected, nodes):
    values = torch.zeros(nodes, dtype=torch.bool)
    for node in range(nodes):
        if (selected >> node) & 1:
            values[node] = True
    return values


def is_independent(selected, neighbours):
    chosen = [node for node in range(len(neighbours)) if selected[node]]
    return all(other not in neighbours[node] for i, node in enumerate(chosen) for other in chosen[i + 1:])


def as_mask(selected):
    mask = 0
    for node in range(len(selected)):
        if selected[node]:
            mask |= 1 << node
    return mask


def paired_bootstrap(deltas, rounds, seed=20260101):
    generator = torch.Generator().manual_seed(seed)
    values = torch.tensor(deltas, dtype=torch.float64)
    index = torch.randint(0, values.numel(), (rounds, values.numel()), generator=generator)
    means = values[index].mean(dim=1)
    return float(torch.quantile(means, 0.025)), float(torch.quantile(means, 0.975))


def load_ranker(reference_seed):
    checkpoint = torch.load(pave(reference_seed), map_location="cpu", weights_only=False)
    model = PAVEV3(checkpoint["mode"], checkpoint.get("stats_dim", STATS_DIM))
    model.load_state_dict(checkpoint["model_state"])
    return model.cpu().eval()


def run_pave(graph, selected, choose, seconds):
    deadline = perf_counter() + seconds
    result = platform_search(graph, selected, choose=choose, deadline=deadline, **PAVE_ARGS)
    return result


def main():
    parser = build_parser()
    args = parser.parse_args()
    binary = args.kamis.expanduser().resolve()
    if not binary.is_file():
        parser.error(f"KaMIS binary not found: {binary}")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    output = args.result_dir / f"pave_external_{args.source}"
    output.mkdir(exist_ok=True)

    dataset = dataset_at(args.test_path or str(VALID), torch.float32, args.test_samples,
                         official=args.split == "test")
    ranker = load_ranker(args.reference_seed)
    learner = ranker.choose
    chooser_rng = random.Random(20260101)
    rows = []
    started = perf_counter()
    for index, graph in enumerate(dataset):
        graph_id = f"idx_{index}" if args.split == "test" else Path(dataset.files[index]).stem
        neighbours = neighbours_of(graph)
        nodes = graph.num_nodes
        for seed in args.seeds:
            tick = perf_counter()
            if args.source == "kamis":
                base = kamis(graph, binary, seed, args.kamis_limit)
                source_seconds = perf_counter() - tick
                if not is_independent(base, neighbours):
                    raise ValueError(f"{graph_id}: KaMIS returned a non-independent set")
            else:
                base = minimum_degree(graph)
                source_seconds = perf_counter() - tick
                if not is_independent(base, neighbours):
                    raise ValueError(f"{graph_id}: greedy returned a non-independent set")
            base_size = int(base.sum())
            base_mask = as_mask(base)
            search = SearchGraph(nodes, graph.edge_index, seed=seed)
            row = dict(graph_id=graph_id, seed=seed, source=args.source,
                       kamis_limit=args.kamis_limit, pave_limit=args.pave_limit,
                       kamis_size=base_size, kamis_wall_s=round(source_seconds, 4))

            tick = perf_counter()
            extended = search.extend(base_mask)
            row["extend_gain"] = extended.bit_count() - base_size
            row["kamis_extend_size"] = base_size + row["extend_gain"]
            row["extend_s"] = round(perf_counter() - tick, 5)

            for name, choose in (("kamis_pave", learner),
                                 ("kamis_pave_rule", "coverage"),
                                 ("kamis_pave_random",
                                  lambda g, s, a, r: a[chooser_rng.randrange(len(a))])):
                tick = perf_counter()
                result = run_pave(search, base_mask, choose, args.pave_limit)
                seconds = perf_counter() - tick
                size = result.best.bit_count()
                chosen = as_bool(result.best, nodes)
                if not is_independent(chosen, neighbours):
                    raise ValueError(f"{graph_id}: illegal set in {name}")
                if size < base_size:
                    raise ValueError(f"{graph_id}: {name} shrank the set, expected monotone")
                row[f"{name}_size"] = size
                row[f"{name}_gain"] = size - base_size
                row[f"{name}_s"] = round(seconds, 4)
                row[f"{name}_hit"] = int(result.deadline_hit)
                row[f"{name}_exchanges"] = result.exchanges

            if args.source == "kamis":
                tick = perf_counter()
                long_base = kamis(graph, binary, seed, args.kamis_limit + args.pave_limit)
                row["kamis_long_size"] = int(long_base.sum())
                row["kamis_long_gain"] = row["kamis_long_size"] - base_size
                row["kamis_long_wall_s"] = round(perf_counter() - tick, 4)
                if not is_independent(long_base, neighbours):
                    raise ValueError(f"{graph_id}: KaMIS (long) returned a non-independent set")
            else:
                row["kamis_long_size"] = base_size
                row["kamis_long_gain"] = 0
                row["kamis_long_wall_s"] = 0.0
            rows.append(row)
        if (index + 1) % 20 == 0 or index + 1 == len(dataset):
            print(f"{index + 1}/{len(dataset)} graphs " + " ".join(
                f"{arm}:{mean(row[arm + '_size'] for row in rows):.3f}"
                for arm in ("kamis", "kamis_pave", "kamis_pave_rule", "kamis_pave_random", "kamis_long")),
                flush=True)

    path = output / f"pave_external_{args.source}_ref{args.reference_seed}.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    base_sizes = [row["kamis_size"] for row in rows]
    for arm in ("kamis_extend", "kamis_pave", "kamis_pave_rule", "kamis_pave_random", "kamis_long"):
        sizes = [row[arm + "_size"] for row in rows]
        deltas = [size - base for size, base in zip(sizes, base_sizes)]
        low, high = paired_bootstrap(deltas, args.bootstrap) if len(deltas) > 1 else (0.0, 0.0)
        entry = dict(mean=mean(sizes), delta_mean=mean(deltas), delta_ci=[low, high],
                     wins=sum(1 for delta in deltas if delta > 0),
                     ties=sum(1 for delta in deltas if delta == 0),
                     losses=sum(1 for delta in deltas if delta < 0),
                     graphs=len(deltas))
        if arm in ("kamis_pave", "kamis_pave_rule", "kamis_pave_random"):
            entry["seconds_mean"] = mean(row[arm + "_s"] for row in rows)
            entry["deadline_hit"] = sum(row[arm + "_hit"] for row in rows)
            entry["exchanges_mean"] = mean(row[arm + "_exchanges"] for row in rows)
        if arm == "kamis_long":
            entry["wall_mean_s"] = mean(row["kamis_long_wall_s"] for row in rows)
        summary[arm] = entry
    summary["kamis"] = dict(mean=mean(base_sizes), graphs=len(base_sizes),
                            wall_mean_s=mean(row["kamis_wall_s"] for row in rows))
    summary["extend_gain_mean"] = mean(row["extend_gain"] for row in rows)
    for arm, entry in summary.items():
        if arm in ("kamis", "extend_gain_mean"):
            continue
        print(f"{arm:<18} {entry['mean']:.4f} delta {entry['delta_mean']:+.4f} "
              f"[{entry['delta_ci'][0]:+.4f},{entry['delta_ci'][1]:+.4f}] "
              f"{entry['wins']}/{entry['ties']}/{entry['losses']}", flush=True)

    metadata = dict(version="v8_pave_external", status="completed", mode="pave_external",
                    split=args.split, samples=len(dataset), seeds=args.seeds,
                    kamis=str(binary), kamis_sha256=digest(binary),
                    kamis_limit=args.kamis_limit, pave_limit=args.pave_limit,
                    pave_args=PAVE_ARGS, pave_checkpoint_sha256=digest(pave(args.reference_seed)),
                    wall_time_s=round(perf_counter() - started, 2),
                    summary=summary, config={k: str(v) for k, v in vars(args).items()},
                    **provenance(args.reference_seed))
    (output / f"summary_ref{args.reference_seed}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
