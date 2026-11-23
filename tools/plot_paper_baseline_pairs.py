"""Draw graph-seed paired outcomes for the primary 200 ms comparator cohort."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from compare_mis_runs import compare
from paper_figure_style import (BLUE, GRAY, INK, YELLOW, apply_style, boxed_legend,
                                panel_title, style_axis)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs/v11"
OUT = ROOT / "overleaf/figures/baseline_pairs.pdf"
REF = RUNS / "reference_deadline_20261119"
FOLLOWUP = RUNS / "reference_deadline_20261119_followup"
COMPARATORS = {
    "Released RLNN": FOLLOWUP / "author_rlnn/per_graph.csv",
    "RLSA": REF / "rlsa/per_graph.csv",
    "DISCS iSCO": REF / "isco_full_window/per_graph.csv",
}


def load(paths):
    scores = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row["graph_id"], int(row["seed"]), int(row["repeat"]))
                if key in scores:
                    raise ValueError(f"Repeated graph-seed-replicate in {path}: {key}")
                ready = row["final_ready_ms"]
                scores[key] = (int(row["final_size"]), bool(ready) and float(ready) <= 200)
    return scores


def draw():
    root = RUNS / "fused_pave_02s_deadline"
    pattern = "pave_off_s*/v11_rows.csv"
    fused = load(sorted(root.glob(pattern)))
    apply_style()
    fig, (outcomes, means) = plt.subplots(1, 2, figsize=(6.75, 2.13),
                                          gridspec_kw={"width_ratios": [1.08, 1]})
    for ax in (outcomes, means):
        style_axis(ax, cream=False)
    for x, (name, path) in enumerate(COMPARATORS.items()):
        baseline = load([path])
        if len(fused) != 3000 or fused.keys() != baseline.keys():
            raise ValueError(f"Incomplete pairing for {name}")
        delta = np.array([score - baseline[key][0] for key, (score, ready) in fused.items()])
        both = np.array([ready and baseline[key][1] for key, (score, ready) in fused.items()])
        for offset, condition, color in zip((-.27, 0, .27), (delta > 0, delta == 0, delta < 0),
                                            (BLUE, YELLOW, GRAY)):
            fraction = 100 * condition.mean()
            outcomes.bar(x+offset, fraction, width=.245, color=color, edgecolor=INK, lw=.65)
            outcomes.annotate(f"{fraction:.1f}", (x+offset, fraction), xytext=(0, 2),
                              textcoords="offset points", ha="center", va="bottom", fontsize=6.75)
        summary = compare(root, path, candidate_glob=pattern)
        all_mean, common_mean = delta.mean(), delta[both].mean()
        lo, hi = summary["ci95"]
        means.errorbar(x-.1, all_mean, yerr=[[all_mean-lo],[hi-all_mean]], fmt="o", color=INK,
                       mfc=BLUE, ms=4.8, mew=.65, lw=1.05, capsize=2.5, zorder=4)
        means.scatter(x+.1, common_mean, marker="D", s=19, facecolor="white", edgecolor=INK,
                      linewidth=.8, zorder=3)
        means.annotate(f"{all_mean:.3f}", (x-.1, hi), xytext=(0, 3), textcoords="offset points",
                       va="bottom", ha="center", fontsize=6.8)
        low_point = common_mean < .15
        means.annotate(f"{common_mean:.3f}", (x+.1, common_mean), xytext=(5, 0) if low_point else (0, -5),
                       textcoords="offset points", va="center" if low_point else "top",
                       ha="left" if low_point else "center", fontsize=6.8)
    for ax in (outcomes, means):
        ax.set(xlim=(-.5, 2.5), xticks=range(3), xticklabels=list(COMPARATORS))
    outcomes.set(ylim=(0, 100), yticks=[0, 25, 50, 75, 100], ylabel="Graph–seed pairs (%)")
    means.set(ylim=(-.105, 1.10), yticks=[0, .25, .50, .75, 1.0],
              ylabel="Size difference (vertices)")
    means.axhline(0, lw=.7, color=INK)
    panel_title(outcomes, "a", "Particle outcomes by comparator")
    panel_title(means, "b", "Mean paired size gain")
    boxed_legend(outcomes, handles=[Patch(facecolor=c, edgecolor=INK, lw=.65, label=t) for c, t in
                                    ((BLUE, "Larger"), (YELLOW, "Equal"), (GRAY, "Smaller"))],
                 loc="upper center", bbox_to_anchor=(.5, -.20), ncol=3, handlelength=1)
    boxed_legend(means, handles=[
        Line2D([], [], color=INK, marker="o", mfc=BLUE, mew=.6, ls="-", lw=.8,
               label="All pairs (95% CI)", markersize=4),
        Line2D([], [], markeredgecolor=INK, markerfacecolor="white", marker="D",
               ls="", label="Both on time", markersize=4)],
        loc="upper center", bbox_to_anchor=(.5, -.20), ncol=2, handlelength=1)
    fig.subplots_adjust(left=.083, right=.992, bottom=.25, top=.87, wspace=.38)
    fig.savefig(OUT, metadata={"Title": "Paired particle-search outcomes at 200 ms"})
    plt.close(fig)


if __name__ == "__main__":
    draw()
