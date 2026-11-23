"""Rebuild the manuscript's vector figures from the saved paired experiments."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import networkx as nx
import numpy as np
from paper_figure_style import (BLUE, CREAM, GOLD_LINE, GRAY, GREEN, GRID, INK, MUTED, SKY,
                                YELLOW, apply_style, boxed_legend, panel_title, style_axis)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/v11/paper_ablations"
FIGURES = ROOT / "overleaf/figures"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf", metadata={"Title": name.replace("_", " "), "Author": ""})
    plt.close(fig)


def execution_effects():
    cross = read(DATA / "cuda_graph_compile_20261119/summary.json")
    rows = []
    for name, key in (("Replay + compile", "full_vs_eager_no_compile"),
                      ("Replay alone", "capture_without_compile"),
                      ("Compile in eager", "compile_under_eager"),
                      ("Compile with replay", "compile_under_capture")):
        fixed = cross["fixed130"]["effects"][key]
        timed = cross["deadline200"]["effects"][key]
        rows.append((name, -fixed["wall_ms"]["mean"], timed["steps"]["mean"],
                     fixed["size"], timed["size"]))
    for name, experiment in (("Dense vs. CSR", "adjacency_20261119"),
                             ("FP16 vs. FP32", "precision_20261119")):
        fixed, timed = [read(DATA / experiment / mode / "summary.json")
                        for mode in ("fixed130", "deadline200")]
        effects = [{"mean": result["mean_paired_delta"], "ci95": result["ci95"]}
                   for result in (fixed, timed)]
        rows.append((name, fixed["reference"]["mean_wall_ms"] - fixed["candidate"]["mean_wall_ms"],
                     timed["candidate"]["mean_steps"] - timed["reference"]["mean_steps"], *effects))

    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.50),
                             gridspec_kw={"width_ratios": [1, 1.02, 1.21]})
    fig.subplots_adjust(left=.173, right=.983, bottom=.28, top=.87, wspace=.31)
    for ax in axes:
        style_axis(ax, grid="x", cream=False)
        ax.set_ylim(-.5, len(rows)-.46)
        ax.set_yticks(range(len(rows)))
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.axvline(0, color=MUTED, linewidth=.7, zorder=1)
    axes[0].set_yticklabels([row[0] for row in rows[::-1]])
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])
    panel_title(axes[0], "a", "Time at 130 steps")
    panel_title(axes[1], "b", "Search at 200 ms")
    panel_title(axes[2], "c", "Feasible size gain")
    axes[0].set(xlim=(-10, 275), xticks=[0, 80, 160, 240], xlabel="Time saved (ms)")
    axes[1].set(xlim=(-20, 355), xticks=[0, 100, 200, 300], xlabel="Additional steps")
    axes[2].set(xlim=(-.041, .38), xticks=[0,.1,.2,.3], xlabel="Additional vertices")
    for y, (_, ms, steps, fixed, timed) in enumerate(rows[::-1]):
        for ax, value, fill in ((axes[0], ms, GRAY), (axes[1], steps, SKY)):
            ax.barh(y, value, left=0, height=.56, color=YELLOW if y == len(rows)-1 else fill,
                    edgecolor=INK, lw=.62, zorder=3)
            ax.annotate(f"{value:.1f}".replace("-", "−"), (max(value, 0), y), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=6.65,
                        bbox=dict(fc="white", ec="none", pad=.12))
        for result, offset, color, marker, fill in (
                (fixed, .16, MUTED, "s", "white"), (timed, -.16, BLUE, "o", BLUE)):
            mean = result["mean"]
            lo, hi = result["ci95"]
            axes[2].errorbar(mean,y+offset,xerr=[[mean-lo],[hi-mean]],fmt=marker,
                             color=color, mec=INK, mfc=fill, ms=4.1, capsize=2.3, lw=1.1,
                             markeredgewidth=.65)
    handles = [Line2D([], [], color=MUTED, marker="s", mec=INK, mfc="white", lw=1, ms=4,
                      mew=.65, label="130 steps"),
               Line2D([], [], color=BLUE, marker="o", mec=INK, mfc=BLUE, lw=1, ms=4,
                      mew=.65, label="200 ms")]
    boxed_legend(axes[2], handles=handles, bbox_to_anchor=(.5, -.30), loc="upper center",
                 borderaxespad=0, ncol=2, handlelength=1.2)
    save(fig,"execution_effects")


def decode_times(arm):
    times = []
    for file in sorted((DATA / "decode_schedule_20261119" / arm).rglob("v11_rows.csv")):
        with file.open(newline="", encoding="utf-8") as handle:
            times.extend(float(row["final_ready_ms"]) for row in csv.DictReader(handle))
    if len(times) != 3000 or max(times) > 200:
        raise ValueError(f"Incomplete deadline sample for {arm}")
    return np.sort(times)


def membership(ax, checkpoints):
    steps = np.array([row["steps"] for row in checkpoints])
    half_stride = np.diff(steps)[0] / 2
    states = [set(row["nodes"]) for row in checkpoints]
    vertices = sorted(set.union(*states), key=lambda v: (next(i for i, s in enumerate(states) if v in s), v))
    ax.set_facecolor(CREAM)
    for index, vertex in enumerate(vertices):
        active = np.array([vertex in state for state in states])
        changes = np.diff(np.r_[False, active, False].astype(int))
        for start, stop in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            left = steps[start] - half_stride
            right = steps[stop - 1] + half_stride
            ax.add_patch(Rectangle((left, index - .48), right - left, .96,
                                   color=GREEN if vertex in states[-1] else "#75787D", lw=0))
    ax.set(xlim=(steps[0] - half_stride, steps[-1] + half_stride),
           ylim=(len(vertices) - .5, -.5), yticks=[], xticks=[0, 100, 200, steps[-1]],
           xlabel="Search steps", ylabel=f"{len(vertices)} vertices")
    ax.grid(axis="x", color=GRID, linewidth=.5, linestyle=":")
    ax.set_axisbelow(True)


def endpoint_graph(ax, record):
    first, last = [set(record["checkpoints"][i]["nodes"]) for i in (0, -1)]
    graph = nx.Graph()
    graph.add_nodes_from(first | last)
    graph.add_edges_from((u,v) for u,v in record["edges"] if u in graph and v in graph)
    parts = sorted(nx.connected_components(graph), key=lambda part: (-len(part), min(part)))
    columns = 4
    for index, part in enumerate(parts):
        cx, cy = index % columns, -(index // columns)
        sub = graph.subgraph(part)
        positions = {}
        if len(part) == 1:
            positions[min(part)] = (cx,cy)
        else:
            for side, offset in ((first, -.24), (last, .24)):
                side_nodes = sorted(part & side)
                heights = np.linspace(-.24,.24,len(side_nodes)) if len(side_nodes)>1 else [0]
                positions.update({v:(cx+offset,cy+h) for v,h in zip(side_nodes,heights)})
        for u, v in sub.edges:
            ax.plot(*zip(positions[u], positions[v]), color=INK, lw=.75, zorder=1)
        for v, (x, y) in positions.items():
            fill, marker = ((YELLOW, "s") if v in first & last else
                            (BLUE, "^") if v in first else (GREEN, "o"))
            ax.scatter(x, y, marker=marker, s=20, facecolors=fill, edgecolors=INK,
                       linewidths=.45, zorder=2)
    ax.set(xlim=(-.55, columns - .45), ylim=(-4.55, .55), xticks=[], yticks=[])
    ax.set_facecolor(CREAM)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(GRID)
        spine.set_linewidth(.55)
    return [Line2D([], [], marker=marker, mfc=fill, mec=INK, mew=.5, lw=0, ms=4.5, label=label)
            for marker, fill, label in (("^", BLUE, f"Initial only ({len(first - last)})"),
                                         ("o", GREEN, f"Final only ({len(last - first)})"),
                                         ("s", YELLOW, f"Shared ({len(first & last)})"))]


def decode_cases():
    record = read(DATA / "qual_instance_20261120/case.json")
    fig, (clock, trace, graph) = plt.subplots(1, 3, figsize=(6.75, 2.67),
                                  gridspec_kw={"width_ratios": [1.13, 1, 1]})
    fig.subplots_adjust(left=.076, right=.992, bottom=.345, top=.86, wspace=.29)
    panel_title(clock, "a", "Host availability")
    panel_title(trace, "b", "Vertex membership")
    panel_title(graph, "c", "Endpoint conflicts")
    clock.set_facecolor(CREAM)
    handles = []
    for arm, stroke, fill, style, marker, label in (
            ("full", GOLD_LINE, YELLOW, "-", "o", "Two decodes"),
            ("terminal", MUTED, GRAY, "--", "s", "Terminal only")):
        times = decode_times(arm)
        fraction = np.arange(1, len(times) + 1) / len(times)
        clock.step(np.r_[0, times, 200], np.r_[0, fraction, 1], where="post",
                   color=stroke, ls=style, lw=1.5)
        mid = np.median(times)
        clock.plot(mid, .5, marker=marker, ls="", mfc=fill, mec=INK, mew=.55, ms=4.2)
        handles.append(Line2D([], [], color=stroke, ls=style, marker=marker, mfc=fill, mec=INK,
                              mew=.5, lw=1.2, ms=3.8, label=f"{label} ({mid:.1f} ms)"))
    clock.set(xlim=(0, 205), ylim=(-.035, 1.04), yticks=[0, .5, 1], yticklabels=["0", "0.5", "1"],
              xticks=[0, 50, 100, 150, 200], xlabel="Elapsed solve time (ms)", ylabel="Fraction of runs")
    clock.grid(color=GRID, lw=.5)
    membership(trace, record["checkpoints"])
    graph_handles = endpoint_graph(graph, record)
    trace_handles = [Rectangle((0, 0), 1, 1, fc=color, ec=INK, lw=.45, label=name)
                     for color, name in ((GREEN, "In terminal set"), ("#75787D", "Other vertices"))]
    for axis, legend_handles in ((clock, handles), (trace, trace_handles), (graph, graph_handles)):
        fig.legend(handles=legend_handles, loc="upper left", frameon=False,
                   bbox_to_anchor=(axis.get_position().x0, .155), fontsize=6.35, borderaxespad=0,
                   borderpad=0, handlelength=1.25, handletextpad=.45, labelspacing=.18)
    save(fig, "decode_cases")


def solve_clock():
    schedules = (("full", "Two decodes"), ("terminal", "Terminal only"))
    fig, (clock, metrics) = plt.subplots(1, 2, figsize=(6.75, 2.14),
                                         gridspec_kw={"width_ratios": [1.53, 1]})
    fig.subplots_adjust(left=.125, right=.992, top=.84, bottom=.36, wspace=.16)
    panel_title(clock, "a", "When the solution reaches the host")
    panel_title(metrics, "b", "Per-graph measurements")
    clock.set(xlim=(-3, 207), ylim=(-.42, 1.60), xticks=[0, 50, 100, 150, 200],
              yticks=[1, 0], yticklabels=[name for _, name in schedules],
              xlabel="Elapsed solve time (ms)")
    clock.set_facecolor(CREAM)
    clock.tick_params(axis="y", length=0)
    clock.axvline(200, color=INK, ls=(0, (3, 2)), lw=.8)
    clock.grid(axis="x", color=GRID, lw=.5)
    metrics.set(xlim=(0, 1), ylim=clock.get_ylim())
    metrics.set_axis_off()
    columns = (.105, .323, .587, .861)
    metrics.add_patch(Rectangle((0, 1.24), 1, .37, fc="#F5EEE5", ec="none"))
    metrics.hlines([1.24, .54, -.33], 0, 1, color=GRID, lw=.7)
    for x, label in zip(columns, ("Steps", "Size", "Ready", "Return")):
        metrics.text(x, 1.425, label, ha="center", va="center", fontsize=7.1, weight="bold")
    for (folder, _), y in zip(schedules, (1, 0)):
        run = DATA / "decode_schedule_20261119" / folder / "batch_1017/seed_1017/v11_rows.csv"
        with run.open(encoding="utf-8", newline="") as stream:
            record = next(row for row in csv.DictReader(stream) if row["graph_id"] == "idx_0")
        first, last, wall = (float(record[k]) for k in ("final_ready_ms", "last_decode_ready_ms", "wall_ms"))
        clock.plot([0, wall], [y + .09] * 2, color=BLUE, lw=1.35, solid_capstyle="butt")
        clock.plot([first, 200], [y - .12] * 2, color=GREEN, lw=3, solid_capstyle="butt")
        clock.plot(first, y - .12, marker="o", mec=INK, mfc=GREEN, mew=.5, ms=3.6)
        for readout in sorted({first, last}):
            clock.plot(readout, y + .09, marker="D", mec=INK, mfc=YELLOW, mew=.55, ms=4.3)
            clock.annotate(f"{readout:.1f}", (readout, y + .09), xytext=(-2, 4.5),
                           textcoords="offset points", ha="right", va="bottom", fontsize=6.4)
        for x, value in zip(columns, (record["steps"], record["final_size"], f"{first:.1f}", f"{wall:.1f}")):
            metrics.text(x, y, value, ha="center", va="center", fontsize=8.1)
    metrics.text(.5, -.71, "Ready and return times in ms", ha="center", fontsize=6.7)
    handles = [Line2D([], [], color=BLUE, lw=1.35, label="Solver active"),
               Line2D([], [], marker="D", mec=INK, mfc=YELLOW, mew=.55, lw=0, ms=4.3, label="Decode complete"),
               Line2D([], [], color=GREEN, lw=3, marker="o", mec=INK, mfc=GREEN, mew=.5, ms=3.6,
                      label="Scored set available"),
               Line2D([], [], color=INK, ls=(0, (3, 2)), lw=.8, label="200 ms deadline")]
    legend = fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(.5, .015),
                        handlelength=1.4, columnspacing=1.2, fontsize=6.7)
    legend.get_frame().set_linewidth(.55)
    save(fig, "solve_clock")


def main():
    apply_style()
    execution_effects()
    decode_cases()
    solve_clock()


if __name__ == "__main__":
    main()
