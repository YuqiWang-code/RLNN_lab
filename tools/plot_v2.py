"""Plot the measured RB200–300 quality and solve time by evaluation protocol."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CORE = {
    ("RLNN", "none"): ("RLNN", "#2965A3", "o"),
    ("RLNN", "rule23"): ("RLNN + rule exchanges", "#2965A3", "D"),
    ("SEGF", "none"): ("SEGF", "#B54C35", "o"),
    ("SEGF", "rule23"): ("SEGF + rule exchanges", "#B54C35", "D"),
    ("RLNN-tuned", "none"): ("RLNN tuned", "#63725D", "^"),
}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "outputs/v2/pareto.svg")
    args = parser.parse_args()
    groups = defaultdict(list)
    for path in sorted((root / "outputs/v2").glob("*/*/run.json")):
        if path.parent.parent.name.startswith("smoke_"):
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["mode"] == "eval" and row["split"] == "test" and row["samples"] == 1000:
            key = row["config"]["time_budget_s"], row["method"], row["post"]
            groups[key].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.35), sharey=True, layout="constrained")
    for ax, budget, title in zip(axes, (None, 0.5), ("Fixed search · K=20, T=100", "Fixed time · 0.5 s budget")):
        measured = []
        for (setting, method, post), group in groups.items():
            if setting != budget:
                continue
            x = mean(row["mean_solve_time_s"] for row in group)
            y = mean(row["mean_size"] for row in group)
            label, color, marker = CORE.get((method, post), ("Other controls", "#A1AAB4", "."))
            ax.scatter(x, y, color=color, marker=marker, s=55 if marker != "." else 36,
                       label=label, zorder=3 if marker != "." else 2)
            measured.append((x, y))
        frontier = []
        for point in sorted(measured, key=lambda point: (point[0], -point[1])):
            if not frontier or point[1] > frontier[-1][1]:
                frontier.append(point)
        if len(frontier) > 1:
            ax.plot(*zip(*frontier), color="#4A5568", linewidth=1, linestyle=":", label="Measured Pareto front")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("Mean solve time (s / graph)")
        ax.grid(True, color="#E7EBF0", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean independent set size")
    legend = {}
    for ax in axes:
        handles, names = ax.get_legend_handles_labels()
        legend.update(zip(names, handles))
    fig.legend(legend.values(), legend.keys(), loc="outside lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("RLNN v2 on 1,000 RB200–300 test graphs", x=0.03, ha="left", fontsize=13)
    fig.savefig(args.output, format="svg")


if __name__ == "__main__":
    main()
