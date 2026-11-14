#!/usr/bin/env python3
"""Paired comparison between two v8 evaluation CSVs.

The v8 harnesses write one row per (graph, budget, cell, seed, repeat).  When a
backbone or a head changes we need "same graphs, same seeds, how much did the
solution move" rather than two independent means.  This script keys the rows by
graph and prints a figure-level paired bootstrap.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np


def read_sizes(path: Path, cell: str, budget: float, key: str = "final_size"):
    table: dict[tuple[str, str], list[float]] = defaultdict(list)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row.get("cell") != cell:
                continue
            if abs(float(row["budget"]) - budget) > 1e-9:
                continue
            table[(row["graph_id"], row.get("repeat", "0"))].append(float(row[key]))
    return {k: mean(v) for k, v in table.items()}


def paired(base, test, resamples: int = 5000, seed: int = 0):
    keys = sorted(set(base) & set(test))
    if not keys:
        raise SystemExit("no shared graphs")
    diffs = np.array([test[k] - base[k] for k in keys], dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = generator.choice(diffs, size=(resamples, len(diffs)), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    wins = int((diffs > 0).sum())
    ties = int((diffs == 0).sum())
    losses = int((diffs < 0).sum())
    return diffs.mean(), low, high, wins, ties, losses, len(keys)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--cells", nargs="+", default=["captured/cached"])
    parser.add_argument("--key", default="final_size", choices=["final_size", "search_size"])
    parser.add_argument("--label", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tag = f"{args.label} " if args.label else ""
    print(f"{tag}base {args.base}")
    print(f"{tag}test {args.test}")
    for budget in args.budgets:
        for cell in args.cells:
            base = read_sizes(args.base, cell, budget, args.key)
            test = read_sizes(args.test, cell, budget, args.key)
            centre, low, high, wins, ties, losses, count = paired(base, test)
            print(f"{budget:.2f}s {cell:16s} {args.key} "
                  f"base {mean(base.values()):.4f} test {mean(test.values()):.4f} "
                  f"diff {centre:+.4f} [{low:+.4f}, {high:+.4f}] "
                  f"{wins}/{ties}/{losses} over {count} graphs")


if __name__ == "__main__":
    main()
