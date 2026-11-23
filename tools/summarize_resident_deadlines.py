"""Summarize independently budgeted MIS runs on the released RB and ER graphs."""

import argparse
import csv
import json
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt

from compare_mis_runs import compare, read_rows, summarize
from paper_figure_style import (BLUE, GRAY, GREEN, INK, MUTED, ORANGE, apply_style,
                                boxed_legend, panel_title, style_axis)


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs/v16/anytime_author"
SPLITS = {"RB_small": (1000, "RB-[200–300]"), "RB_large": (1000, "RB-[800–1200]"),
          "ER_small": (128, "ER-[700–800]"), "ER_large": (16, "ER-[9000–11000]")}
BUDGETS = (50, 100, 200, 500)
METHODS = ("learned", "degree", "online_mis")
SEEDS = (1017, 1019, 1021)


def cell_stats(path, graph_count, budget, optima=None):
    with path.open(encoding="utf-8", newline="") as stream:
        raw = list(csv.DictReader(stream))
    keys = {(r["graph_id"], int(r["seed"])) for r in raw}
    expected = {(f"idx_{i}", seed) for i in range(graph_count) for seed in SEEDS}
    if keys != expected or len(raw) != len(expected):
        raise ValueError(f"Incomplete or duplicated cohort: {path}")
    first_times = []
    optimal_pairs = 0
    for row in raw:
        if abs(1000 * float(row["budget"]) - budget) > 1e-8:
            raise ValueError(f"Wrong deadline: {path}")
        first, last = row["first_ready_ms"], row["final_ready_ms"]
        if bool(first) != bool(last) or (first and not 0 <= float(first) <= float(last) <= budget):
            raise ValueError(f"Invalid delivery window: {path}")
        if not first and int(row["final_size"]) != 0:
            raise ValueError(f"Scored set with no delivery: {path}")
        if first:
            first_times.append(float(first))
            if optima is not None:
                size = int(row["final_size"])
                optimum = optima[row["graph_id"]]
                if size > optimum:
                    raise ValueError(f"Size exceeds the verified optimum: {path}, {row['graph_id']}")
                optimal_pairs += size == optimum
    stats = summarize(read_rows(path), budget)
    return dict(graphs=graph_count, pairs=len(raw), mean_size=stats["mean_size"], mean_steps=stats["mean_steps"],
                mean_wall_ms=stats["mean_wall_ms"], p95_wall_ms=stats["wall_p95_ms"],
                ready=stats["ready_by_deadline"], returned=sum(float(r["wall_ms"]) <= budget for r in raw),
                first_ready_p50_ms=median(first_times) if first_times else None,
                optimal_pairs=optimal_pairs if optima is not None else None)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, paired):
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(6.85, 3.7))
    fig.subplots_adjust(left=.082, right=.985, bottom=.13, top=.85, hspace=.6, wspace=.49)
    styles = {"degree": ("Degree priority", GREEN, "--", "s", 6.4),
              "learned": ("RLNN priority", BLUE, "-", "o", 3.6),
              "online_mis": ("Resident OnlineMIS", ORANGE, "-.", "^", 4.5)}
    for ax, letter, (split, (_, title)) in zip(axes.flat, "ABCD", SPLITS.items()):
        style_axis(ax, grid="both")
        for method, (label, color, line, marker, size) in styles.items():
            values = {int(r["budget_ms"]): float(r["mean_size"]) for r in rows
                      if r["split"] == split and r["method"] == method}
            ax.plot(BUDGETS, [values[b] for b in BUDGETS], color=color, ls=line, lw=1.45,
                    marker=marker, markersize=size, markeredgewidth=.55,
                    markerfacecolor=color, markeredgecolor=INK, label=label)
        panel_title(ax, letter, title)
        if ax in axes[:, 0]:
            ax.set_ylabel("Mean feasible size")
        ax.ticklabel_format(axis="y", useOffset=False)
    axes[0, 0].set(ylim=(20.013, 20.128), yticks=[20.04, 20.08, 20.12])
    certified = next(float(r["mean_size"]) for r in rows
                     if r["split"] == "RB_small" and r["method"] == "learned")
    axes[0, 0].text(.07, .73, f"Both priorities: {certified:.3f}", transform=axes[0, 0].transAxes,
                    color=INK, fontsize=7.0)
    ax = axes[1, 1]
    panel_title(ax, "E", "Small ER gain")
    style_axis(ax, cream=False)
    ax.axhline(0, color=MUTED, lw=.75)
    batch = {int(r["budget_ms"]): [float(r[k]) for k in ("mean_gain", "ci_low", "ci_high")]
             for r in paired if r["split"] == "ER_small" and r["reference"] == "degree"}
    repeat = ROOT / "outputs/v16/er_interleaved/paired.csv"
    with repeat.open(encoding="utf-8", newline="") as stream:
        interleaved = {int(r["budget_ms"]): [float(r[k]) for k in ("mean_paired_delta", "ci95_low", "ci95_high")]
                       for r in csv.DictReader(stream)
                       if r["candidate"] == "learned" and r["reference"] == "degree"}
    for values, offset, color, marker, face, label in (
            (batch, .945, MUTED, "s", GRAY, "Separate batches"),
            (interleaved, 1.056, BLUE, "o", BLUE, "One-core repeat")):
        estimates = [values[b][0] for b in BUDGETS]
        errors = [[values[b][0] - values[b][1] for b in BUDGETS],
                  [values[b][2] - values[b][0] for b in BUDGETS]]
        ax.errorbar([b * offset for b in BUDGETS], estimates, yerr=errors, color=color,
                    fmt=marker, ms=4.0, mfc=face, mec=INK, mew=.55, capsize=2.2, elinewidth=1,
                    label=label)
    boxed_legend(ax, loc="lower right", fontsize=5.8, handlelength=.8)
    ax.set(ylim=(-.69, .25), yticks=[-.6, -.3, 0])
    ax = axes[1, 2]
    panel_title(ax, "F", "Large ER gain")
    style_axis(ax, cream=False)
    ax.axhline(0, color=MUTED, lw=.75)
    gains = sorted((r for r in paired if r["split"] == "ER_large" and r["reference"] == "degree"),
                   key=lambda r: int(r["budget_ms"]))
    centers = [float(r["mean_gain"]) for r in gains]
    errors = [[v - float(r["ci_low"]) for v, r in zip(centers, gains)],
              [float(r["ci_high"]) - v for v, r in zip(centers, gains)]]
    ax.errorbar(BUDGETS, centers, yerr=errors, fmt="s", color=MUTED, mfc=GRAY, mec=INK,
                ms=4, mew=.55, capsize=2.2, elinewidth=1)
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_xlim(41, 610)
        ax.set_xticks(BUDGETS, labels=[str(b) for b in BUDGETS])
        ax.minorticks_off()
    for ax in axes[1]:
        ax.set_xlabel("Deadline (ms)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = (1, 0, 2)
    legend = fig.legend([handles[i] for i in order], [labels[i] for i in order],
                        loc="upper center", bbox_to_anchor=(.5, .998), ncol=3, handlelength=2.1,
                        columnspacing=1.9)
    legend.get_frame().set_linewidth(.55)
    fig.savefig(ROOT / "overleaf/figures/resident_deadlines.pdf",
                metadata={"Title": "Resident MIS search by deadline"})
    plt.close(fig)


def paper_table(rows):
    by_cell = {(r["split"], r["budget_ms"], r["method"]): r for r in rows}
    lines = [r"\begin{table*}[t]", r"\centering\scriptsize\setlength{\tabcolsep}{4.8pt}",
             r"\caption{Resident solvers at four independently run deadlines. Size averages the largest feasible set on the host by the cutoff over all graph--seed pairs. Wall is mean return time in milliseconds. Ready counts runs with a timely host incumbent, and Ret.\ counts solves that finished by the deadline. Each RB set has 3,000 pairs per method and budget. The small and large ER sets have 384 and 48, respectively.}",
             r"\label{tab:resident-details}", r"\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrrrrrrrrr}",
             r"\toprule", r"\paperHeader & & \multicolumn{4}{c}{\cellcolor{TableBlue}\paperHead{RLNN priority}} & \multicolumn{4}{c}{\cellcolor{TableGreen}\paperHead{Degree priority}} & \multicolumn{4}{c}{\cellcolor{TableOrange}\paperHead{Resident OnlineMIS}} \\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}",
             r"\paperHeader \paperHead{Graph set} & \paperHead{Budget (ms)}" + r" & \paperHead{Size\,$\uparrow$} & \paperHead{Wall\,$\downarrow$} & \paperHead{Ready} & \paperHead{Ret.}" * 3 + r" \\"]
    for split, (_, title) in SPLITS.items():
        lines.append(r"\midrule")
        for budget in BUDGETS:
            if budget == 200:
                lines.append(r"\rowcolor{TableGold}")
            items = [title.replace("–", "--") if budget == BUDGETS[0] else "", str(budget)]
            for method in METHODS:
                row = by_cell[split, budget, method]
                items.extend((f"{row['mean_size']:.4f}", f"{row['mean_wall_ms']:.2f}",
                              str(row["ready"]), str(row["returned"])))
            lines.append(" & ".join(items) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}}", r"\end{table*}"))
    (ROOT / "overleaf/sections/resident_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_cohort(base, split, count, budget, optima=None):
    summaries = [
        dict(split=split, budget_ms=budget, method=method,
             **cell_stats(base / method / "per_graph.csv", count, budget, optima))
        for method in METHODS
    ]
    effects = []
    for reference in ("degree", "online_mis"):
        delta = compare(base / "learned/per_graph.csv", base / reference / "per_graph.csv", budget)
        effects.append(dict(split=split, budget_ms=budget, method="learned", reference=reference,
                            graphs=count, pairs=delta["pairs"], mean_gain=delta["mean_paired_delta"],
                            ci_low=delta["ci95"][0], ci_high=delta["ci95"][1],
                            wins=delta["wins"], ties=delta["ties"], losses=delta["losses"]))
    return summaries, effects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, help="summarize one author-test directory with all three resident arms")
    base = parser.parse_args().cohort
    if base is not None:
        run = json.loads((base / "learned/run.json").read_text(encoding="utf-8"))
        split = Path(run["dataset"]).name.removesuffix("_author_test")
        summaries, effects = summarize_cohort(base, split, run["graphs"], run["budget_ms"])
        write_csv(base / "summary.csv", summaries)
        write_csv(base / "paired.csv", effects)
        return

    summaries, effects = [], []
    proof = ROOT / "outputs/v15/priority_200ms_20261121/offline_optimality/RB_small/optima.csv"
    with proof.open(encoding="utf-8", newline="") as stream:
        small_rb_optima = {r["graph_id"]: int(r["optimum"]) for r in csv.DictReader(stream)}
    for split, (count, _) in SPLITS.items():
        for budget in BUDGETS:
            base = RUNS / split / str(budget)
            if split == "ER_large" and budget == 200:
                base = ROOT / "outputs/v16/er_large_200ms"
            cohort, contrasts = summarize_cohort(base, split, count, budget, small_rb_optima if split == "RB_small" else None)
            summaries.extend(cohort)
            effects.extend(contrasts)
    write_csv(RUNS / "summary.csv", summaries)
    write_csv(RUNS / "paired.csv", effects)
    plot(summaries, effects)
    paper_table(summaries)


if __name__ == "__main__":
    main()
