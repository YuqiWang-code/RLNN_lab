"""Two-component decomposition for the registered paper claim.

For every (clock, cell) the CSV already carries both numbers we need: the
incumbent *before* the platform runs (``search_size``) and *after*
(``final_size``).  Turning that into the 2x2 of the claim -- capture x platform --
tells us whether the two components add up or merely substitute for one another.

Usage::

    python models/summarize_v8_claim.py --csv outputs/v8_macro/author_test/capture_budget/capture_budget_ref0.csv
"""
import argparse
import csv
from pathlib import Path
from statistics import mean

import torch

BOOTSTRAP = 5000
BASE = "eager/eager"


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def graph_means(rows):
    """``{(budget, cell): {graph_id: (search, final, steps)}}``."""
    buckets = {}
    for row in rows:
        key = (float(row["budget"]), row["cell"])
        item = buckets.setdefault(key, {}).setdefault(row["graph_id"], [[], [], []])
        item[0].append(float(row["search_size"]))
        item[1].append(float(row["final_size"]))
        item[2].append(float(row["steps"]))
    return {key: {graph: (mean(a), mean(b), mean(c)) for graph, (a, b, c) in per_graph.items()}
            for key, per_graph in buckets.items()}


def bootstrap(values, resamples=BOOTSTRAP, seed=0):
    tensor = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    index = torch.randint(0, tensor.numel(), (resamples, tensor.numel()), generator=generator)
    means = tensor[index].mean(dim=1)
    return dict(mean=float(tensor.mean()), low=float(means.quantile(0.025)),
                high=float(means.quantile(0.975)), wins=sum(v > 0 for v in values),
                ties=sum(v == 0 for v in values), losses=sum(v < 0 for v in values), n=len(values))


def paired(left, right, column):
    shared = sorted(set(left) & set(right))
    diffs = [left[graph][column] - right[graph][column] for graph in shared]
    return bootstrap(diffs)


def cell_text(stats):
    if stats is None:
        return "-"
    return (f"{stats['mean']:+.4f} [{stats['low']:+.4f},{stats['high']:+.4f}] "
            f"{stats['wins']}/{stats['ties']}/{stats['losses']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    table = graph_means(read_rows(args.csv))
    budgets = sorted({budget for budget, _ in table})
    cells = sorted({cell for _, cell in table})
    order = [cell for cell in (BASE, "eager/captured", "captured/eager",
                               "captured/captured", "captured/cached") if cell in cells]
    order += [cell for cell in cells if cell not in order]

    print(f"## {args.label or args.csv}")
    print()
    print("| clock | cell | search (no platform) | final (+platform) | platform gain | steps | final vs "
          f"{BASE} | search vs {BASE} |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for budget in budgets:
        for cell in order:
            per_graph = table.get((budget, cell))
            if not per_graph:
                continue
            search = mean(v[0] for v in per_graph.values())
            final = mean(v[1] for v in per_graph.values())
            steps = mean(v[2] for v in per_graph.values())
            reference = table.get((budget, BASE))
            if cell == BASE or reference is None:
                final_text = search_text = "-"
            else:
                final_text = cell_text(paired(per_graph, reference, 1))
                search_text = cell_text(paired(per_graph, reference, 0))
            print(f"| {budget:.1f} s | {cell} | {search:.4f} | {final:.4f} | "
                  f"{final - search:+.4f} | {steps:.0f} | {final_text} | {search_text} |")

    print()
    print("**Capture x platform (same clock).**  Capture effect is measured on the raw "
          "search, platform effect on the same arm.")
    print()
    print("| clock | capture on search | capture on final | platform on eager | platform on captured |")
    print("| --- | --- | --- | --- | --- |")
    for budget in budgets:
        eager = table.get((budget, BASE))
        captured = table.get((budget, "captured/cached")) or table.get((budget, "captured/eager"))
        if not eager or not captured:
            continue
        capture_search = cell_text(paired(captured, eager, 0))
        capture_final = cell_text(paired(captured, eager, 1))
        platform_eager = mean(v[1] - v[0] for v in eager.values())
        platform_captured = mean(v[1] - v[0] for v in captured.values())
        print(f"| {budget:.1f} s | {capture_search} | {capture_final} | "
              f"{platform_eager:+.4f} | {platform_captured:+.4f} |")


if __name__ == "__main__":
    main()
