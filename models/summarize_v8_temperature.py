"""Summarise a temperature sweep: per-cell means and paired diffs against T = 1.

    python models/summarize_v8_temperature.py outputs/v8_macro/temperature/padded/padded_ref0.csv

The base cell is whatever cell carries temperature 1.0 (named ``padded/cached`` by
``eval_v8_padded.cell_of``); every other cell is compared graph by graph against it
at the same budget, so the diff is a within-graph paired effect.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from statistics import mean  # noqa: E402

from eval_v8_capture_budget import paired_bootstrap  # noqa: E402


def load(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def floats(rows, key):
    return [float(row[key]) for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--base", default=None,
                        help="base cell (default: the one whose temperature is 1.0)")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    rows = load(args.csv)
    cells = sorted({row["cell"] for row in rows})
    base = args.base
    if base is None:
        for row in rows:
            if float(row.get("temperature", 1.0)) == 1.0:
                base = row["cell"]
                break
    if base is None:
        raise SystemExit("no cell with temperature 1.0; pass --base")

    label = args.label or Path(args.csv).stem
    print(f"# {label}\n")
    budgets = sorted({float(row["budget"]) for row in rows})
    for budget in budgets:
        chosen = [row for row in rows if float(row["budget"]) == budget]
        print(f"## {budget:g} s\n")
        print("| cell | graphs | size | steps | ms/step | vs base | W/T/L |")
        print("| --- | --- | --- | --- | --- | --- | --- |")
        base_by_graph = defaultdict(list)
        for row in chosen:
            if row["cell"] == base:
                base_by_graph[row["graph_id"]].append(float(row["final_size"]))
        for cell in cells:
            group = [row for row in chosen if row["cell"] == cell]
            if not group:
                continue
            sizes = floats(group, "final_size")
            record = "-"
            wins = ties = losses = 0
            if cell != base:
                diffs = []
                for row in group:
                    left = float(row["final_size"])
                    right = base_by_graph.get(row["graph_id"])
                    if not right:
                        continue
                    diff = left - mean(right)
                    diffs.append(diff)
                    if diff > 1e-9:
                        wins += 1
                    elif diff < -1e-9:
                        losses += 1
                    else:
                        ties += 1
                if diffs:
                    gap, low, high = paired_bootstrap(diffs)
                    record = f"{gap:+.4f} [{low:+.4f},{high:+.4f}]"
            print(f"| {cell} | {len(group)} | {mean(sizes):.4f} | "
                  f"{mean(floats(group, 'steps')):.1f} | "
                  f"{mean(floats(group, 'ms_per_step')):.3f} | {record} | "
                  f"{'-' if cell == base else f'{wins}/{ties}/{losses}'} |")
        print()


if __name__ == "__main__":
    main()
