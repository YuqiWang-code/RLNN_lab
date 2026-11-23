"""Plot time to a certified optimum and the paired effects of feasible search."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from compare_mis_runs import compare, read_rows
from paper_figure_style import (BLUE, GOLD_LINE, GRAY, GREEN, INK, MUTED, ORANGE, YELLOW,
                                apply_style, boxed_legend, panel_title, style_axis)


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
RESIDENT = OUTPUTS / "v16/anytime_author"


def records(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def optimum_curves(ax):
    proof = OUTPUTS / "v15/priority_200ms_20261121/offline_optimality/RB_small/optima.csv"
    optima = {r["graph_id"]: int(r["optimum"]) for r in records(proof)}
    cohort = {(g, str(s), "0") for g in optima for s in (1017, 1019, 1021)}
    particle = OUTPUTS / "v11/paper_ablations/decode_schedule_20261119"
    runs = (
        ("RLNN priority", RESIDENT / "RB_small/200/learned/per_graph.csv", BLUE, BLUE, "o", "-"),
        ("OnlineMIS", RESIDENT / "RB_small/200/online_mis/per_graph.csv", ORANGE, ORANGE, "^", "--"),
        ("Two decodes", particle / "full", GOLD_LINE, YELLOW, "D", "-"),
        ("Terminal only", particle / "terminal", MUTED, GRAY, "s", "--"),
    )
    marks = np.array([25, 75, 125, 175])
    for label, path, line, fill, marker, style in runs:
        solved = read_rows(path)
        if solved.keys() != cohort:
            raise ValueError(f"{path}: graph--seed pairs differ from the certified cohort")
        times = sorted(ready for (g, _, _), (size, _, ready, _) in solved.items()
                       if size == optima[g] and ready is not None and ready <= 200)
        hit_rate = 100 * np.arange(len(times) + 1) / len(cohort)
        ax.step([0, *times, 200], [*hit_rate, hit_rate[-1]], where="post",
                color=line, linestyle=style, lw=1.55)
        rates = 100 * np.searchsorted(times, marks, side="right") / len(cohort)
        ax.scatter(marks, rates, marker=marker, s=13, facecolor=fill, edgecolor=INK, lw=.45, zorder=4)
        ax.plot([], [], linestyle=style, color=line, marker=marker, mfc=fill, mec=INK, mew=.45,
                ms=3.5, label=label)
        if path == particle / "full":
            rate = rates[marks == 125].item()
            ax.annotate(f"{rate:.1f}% by 125 ms", (125, rate), (72, 63),
                        arrowprops=dict(arrowstyle="-", color=GOLD_LINE, lw=.7, shrinkB=3),
                        fontsize=7, ha="center", va="top")
    ax.set(xlim=(0, 201), ylim=(-3, 105), xticks=[0, 50, 100, 150, 200],
           yticks=[0, 25, 50, 75, 100], xlabel="Elapsed solve time (ms)", ylabel="Runs at verified optimum")
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    boxed_legend(ax, loc="upper center", bbox_to_anchor=(.5, -.32), ncol=2, handlelength=2)


def rank_effects(ax):
    separate = {(r["split"], r["reference"]): r for r in records(RESIDENT / "paired.csv")
                if r["budget_ms"] == "200"}
    blocked = {r["reference"]: r for r in records(OUTPUTS / "v16/er_interleaved/paired.csv")
               if r["budget_ms"] == r["reference_budget_ms"] == "200" and r["candidate"] == "learned"}
    dimacs = OUTPUTS / "v16/dimacs_register"
    for method, label, color, marker, offset in (("online_mis", "vs. OnlineMIS", ORANGE, "^", .16),
                                                 ("degree", "vs. degree", GREEN, "s", -.16)):
        reg = compare(dimacs / "learned/per_graph.csv", dimacs / method / "per_graph.csv")
        er = blocked[method]
        values = [(float(r["mean_gain"]), float(r["ci_low"]), float(r["ci_high"]))
                  for r in (separate["RB_small", method], separate["RB_large", method])]
        large = separate["ER_large", method]
        values.extend([(float(er["mean_paired_delta"]), float(er["ci95_low"]), float(er["ci95_high"])),
                       (float(large["mean_gain"]), float(large["ci_low"]), float(large["ci_high"])),
                       (reg["mean_paired_delta"], *reg["ci95"])])
        center, low, high = np.array(values).T
        y = np.arange(4, -1, -1) + offset
        ax.errorbar(center, y, xerr=np.stack((center - low, high - center)), fmt=marker, ls="none",
                    markersize=4.6, capsize=2.3, elinewidth=1.1, color=INK, mec=INK, mew=.65,
                    mfc=color, label=label)
        for mean, end, level in zip(center, high, y):
            label = f"{mean:+.3f}".replace("-", "−") if mean != 0 else "0"
            ax.annotate(label, (end, level), xytext=(4, 0), textcoords="offset points",
                        fontsize=6.55, va="center")
    ax.axvline(0, lw=.75, color=MUTED, zorder=1)
    ax.set(xlim=(-1, 10.4), ylim=(-.5, 4.55), xticks=[0, 2, 4, 6, 8],
           yticks=[4, 3, 2, 1, 0], yticklabels=["RB [200–300]", "RB [800–1200]", "ER [700–800]", "ER [9000–11000]", "Register"],
           xlabel="Paired size difference (vertices)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    boxed_legend(ax, loc="upper center", bbox_to_anchor=(.5, -.32), ncol=2, handlelength=1)


def main():
    apply_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.75, 2.53),
                                      gridspec_kw={"width_ratios":[1.05,1]})
    fig.subplots_adjust(left=.093, right=.99, bottom=.34, top=.90, wspace=.50)
    style_axis(left, grid="both")
    style_axis(right, grid="x", cream=False)
    optimum_curves(left)
    rank_effects(right)
    panel_title(left, "a", "Time to a verified optimum")
    panel_title(right, "b", "Learned priority at 200 ms")
    fig.savefig(ROOT / "overleaf/figures/intro_evidence.pdf",
                metadata={"Title":"Time to a verified optimum and matched 200 ms quality gains"})
    plt.close(fig)


if __name__ == "__main__":
    main()
