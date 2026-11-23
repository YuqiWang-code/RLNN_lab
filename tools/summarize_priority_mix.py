#!/usr/bin/env python3
"""Summarize the rank-composition ablation at the 200 ms deadline.

Reads per_graph.csv from several arm directories, pairs arms by
(graph_id, seed), averages seeds within a graph, and reports pointwise 95%
intervals from resampling graphs. Also writes a markdown table.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(directory):
    rows = {}
    for row in csv.DictReader((Path(directory) / "per_graph.csv").open()):
        rows[(row["graph_id"], int(row["seed"]))] = int(row["final_size"])
    return rows


def graph_means(rows):
    by_graph = {}
    for (graph, _seed), size in rows.items():
        by_graph.setdefault(graph, []).append(size)
    return {graph: float(np.mean(sizes)) for graph, sizes in by_graph.items()}


def paired(a, b, rng, draws=5000):
    graphs = sorted(set(a) & set(b))
    diffs = np.array([a[g] - b[g] for g in graphs])
    boots = np.array([rng.choice(diffs, len(diffs), replace=True).mean() for _ in range(draws)])
    return (len(graphs), float(diffs.mean()),
            float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--reference", default="degree")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    families = sorted(p.name for p in args.root.iterdir() if p.is_dir() and p.name != "ranks")
    summary, paired_rows = [], []
    for family in families:
        arms = {p.name: load(p) for p in sorted((args.root / family).iterdir()) if p.is_dir()}
        if not arms:
            continue
        means = {name: graph_means(rows) for name, rows in arms.items()}
        for name, rows in arms.items():
            sizes = np.array(list(rows.values()))
            summary.append(dict(family=family, arm=name, graphs=len(means[name]),
                                pairs=len(rows), mean_size=float(sizes.mean()),
                                timely=float((sizes > 0).mean())))
        if args.reference in means:
            for name in means:
                if name == args.reference:
                    continue
                n, delta, low, high = paired(means[name], means[args.reference], rng)
                paired_rows.append(dict(family=family, arm=name, reference=args.reference,
                                        graphs=n, delta=delta, ci_low=low, ci_high=high))

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    if paired_rows:
        with (args.output / "paired.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(paired_rows[0]))
            writer.writeheader()
            writer.writerows(paired_rows)
    (args.output / "summary.json").write_text(json.dumps(dict(summary=summary, paired=paired_rows), indent=2))

    lines = ["| family | arm | graphs | mean size | timely | vs %.3f delta [95%% CI] |" % 0]
    lines[0] = "| family | arm | graphs | mean size | timely | delta vs %s [95%% CI] |" % args.reference
    lines.append("|---|---|---:|---:|---:|---|")
    for row in summary:
        match = next((p for p in paired_rows if p["family"] == row["family"] and p["arm"] == row["arm"]), None)
        delta = "" if match is None else f"{match['delta']:+.4f} [{match['ci_low']:+.4f}, {match['ci_high']:+.4f}]"
        lines.append(f"| {row['family']} | {row['arm']} | {row['graphs']} | {row['mean_size']:.4f} | "
                     f"{100*row['timely']:.1f}% | {delta} |")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
