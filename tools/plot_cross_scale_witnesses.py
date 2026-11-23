"""Compare archived 200 ms independent sets on the first graph in each test split."""

import csv
import json
from collections import Counter
from statistics import fmean

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
import numpy as np

from paper_figure_style import BLUE, GRAY, GRID, INK, YELLOW, apply_style, style_axis
from summarize_scale_deadline import ROOT, SEEDS


GRAPH = "idx_0"
SPLITS = {"RB_small": "RB [200–300]", "RB_large": "RB [800–1200]",
          "ER_small": "ER [700–800]", "ER_large": "ER [9000–11000]"}
METHODS = {
    "priority_fused": "Fused RLNN", "priority_degree": "Degree only", "online_mis_resident": "OnlineMIS (resident)",
    "online_mis_tuned": "OnlineMIS (tuned CLI)", "redumis": "KaMIS ReduMIS", "gurobi": "Gurobi",
    "author_rlnn": "Released RLNN", "fld_eg": "FLD-EG (recon.)", "diffuco": "DiffUCO",
    "ltft": "LTFT", "dimes": "DIMES", "difusco": "DIFUSCO", "dgl_treesearch": "DGL-TreeSearch",
    "intel": "INTEL", "ppo_lwd": "PPO-LwD", "isco_full_window": "DISCS iSCO",
    "rlsa": "RLSA", "isco": "iSCO (300 steps)", "online_mis": "OnlineMIS (fixed CLI)",
    "author_rlnn_matched": "RLNN (matched)",
}
FREQUENCIES = ["#F6F2EC", "#CEDEED", "#82AFD1", BLUE]
GATED = {("RB_large", "diffuco"), ("ER_large", "diffuco"), ("ER_large", "dgl_treesearch")}


def selected_runs(records):
    rows = {}
    for record in records:
        seed = int(record["seed"])
        if record["graph_id"] == GRAPH and seed in SEEDS:
            rows[seed] = record
            if len(rows) == len(SEEDS):
                return rows
    raise ValueError(f"Missing graph/seed records for {GRAPH}")


def load_cases():
    with (ROOT / "outputs/v16/four_scale_200ms.csv").open(encoding="utf-8-sig", newline="") as stream:
        cells = {(r["split"], r["method"]): r for r in csv.DictReader(stream)}
    cases = {}
    for split in SPLITS:
        for method in METHODS:
            cell = cells[split, method]
            if cell["status"] != "measured":
                for seed in SEEDS:
                    cases[split, method, seed] = None, dict(status=cell["status"], archive=cell["source"])
                continue
            base = (ROOT / cell["source"]).parent
            with (base / "per_graph.csv").open(encoding="utf-8-sig", newline="") as stream:
                scores = selected_runs(csv.DictReader(stream))
            with (base / "solutions.jsonl").open(encoding="utf-8") as stream:
                witnesses = selected_runs(map(json.loads, stream))
            for seed, result in scores.items():
                updates = witnesses[seed]["updates"]
                best = updates[-1] if updates else None
                nodes = frozenset(best["nodes"]) if best else frozenset()
                size = int(result["final_size"])
                ready = result["final_ready_ms"]
                if (float(result["budget"]) != .2 or bool(ready) != bool(updates)
                        or size != len(nodes) or (best and (len(best["nodes"]) != size
                        or abs(float(ready) - best["ready_ms"]) > 1e-4 or float(ready) > 200))):
                    raise ValueError(f"Score and witness disagree in {base}, {GRAPH}, seed {seed}")
                result["status"] = "delivered" if best else "miss"
                if not best and (split, method) in GATED and int(result["steps"]) == 0:
                    result["status"] = "budget_gate"
                result["archive"] = (base / "solutions.jsonl").relative_to(ROOT).as_posix()
                cases[split, method, seed] = (nodes, result)
    return cases


def save_records(cases):
    target = ROOT / "outputs/paper_20261120/cross_scale_witnesses.csv"
    fields = ["split", "graph_id", "seed", "method", "status", "size", "shared_with_fused",
              "other_vertices", "ready", "first_ready_ms", "best_ready_ms", "wall_ms", "nodes", "source"]
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (split, method, seed), (nodes, row) in cases.items():
            identity = dict(split=split, graph_id=GRAPH, seed=seed, method=METHODS[method],
                            status=row["status"], source=row["archive"])
            if nodes is None:
                writer.writerow(identity)
                continue
            shared = len(nodes & cases[split, "priority_fused", seed][0])
            writer.writerow(dict(identity,
                                 size=len(nodes), shared_with_fused=shared, other_vertices=len(nodes)-shared,
                                 ready=int(bool(row["final_ready_ms"])), first_ready_ms=row["first_ready_ms"],
                                 best_ready_ms=row["final_ready_ms"], wall_ms=row["wall_ms"],
                                 nodes=json.dumps(sorted(nodes), separators=(",", ":")) if row["final_ready_ms"] else ""))


def draw(cases):
    apply_style()
    fig = plt.figure(figsize=(6.75, 6.25))
    grid = fig.add_gridspec(2, 2, left=.178, right=.99, top=.934,
                            bottom=.145, hspace=.45, wspace=.21)
    y = np.arange(len(METHODS))
    for index, (split, title) in enumerate(SPLITS.items()):
        inset = grid[index // 2, index % 2].subgridspec(1, 2, width_ratios=[1.18, 1], wspace=.12)
        bars = fig.add_subplot(inset[0])
        matrix = fig.add_subplot(inset[1])
        frequency = Counter()
        for (case_split, method, _), (nodes, _) in cases.items():
            if case_split == split and nodes is not None and method not in ("online_mis", "author_rlnn_matched"):
                frequency.update(nodes)
        nodes_shown = sorted(frequency, key=lambda v: (-frequency[v], v))[:8]
        shared, other, notes, selections = [], [], [], []
        for method in METHODS:
            sets = [cases[split, method, seed][0] for seed in SEEDS]
            rows = [cases[split, method, seed][1] for seed in SEEDS]
            if sets[0] is None:
                shared.append(0)
                other.append(0)
                notes.append("N/A")
                selections.append([0] * len(nodes_shown))
                continue
            common = [len(nodes & cases[split, "priority_fused", seed][0]) for nodes, seed in zip(sets, SEEDS)]
            shared.append(fmean(common))
            other.append(fmean(len(nodes)-c for nodes, c in zip(sets, common)))
            hits = sum(bool(r["final_ready_ms"]) for r in rows)
            note = f"{hits}/3" if hits < len(SEEDS) or all(not nodes for nodes in sets) else ""
            if all(r["status"] == "budget_gate" for r in rows):
                note += "*"
            notes.append(note)
            selections.append([sum(v in nodes for nodes in sets) for v in nodes_shown])
        style_axis(bars, grid="x")
        bars.barh(y, shared, height=.63, color=BLUE)
        bars.barh(y, other, left=shared, height=.63, color=YELLOW)
        xmax = {"RB_small": 23, "RB_large": 44, "ER_small": 51, "ER_large": 432}[split]
        bars.set_xlim(0, xmax)
        bars.set_xticks({"RB_small": (0, 10, 20), "RB_large": (0, 20, 40),
                         "ER_small": (0, 20, 40), "ER_large": (0, 200, 400)}[split])
        bars.set_ylim(len(y)-.4, -.6)
        bars.set_yticks(y, list(METHODS.values()) if index % 2 == 0 else [])
        bars.tick_params(axis="y", length=0, pad=2, labelsize=6.4)
        bars.tick_params(axis="x", labelsize=6.2, pad=1)
        bars.set_xlabel("Mean set size", fontsize=6.6, labelpad=1)
        for row, note in enumerate(notes):
            if note:
                bars.text(shared[row] + other[row] + .025*xmax, row, note, ha="left",
                          va="center", fontsize=5.6, color=INK)
            if note == "N/A":
                matrix.add_patch(Rectangle((-.5, row-.5), len(nodes_shown), 1,
                                           facecolor="white", edgecolor=GRAY, hatch="////", linewidth=.35, zorder=3))
        pos = bars.get_position()
        fig.text(pos.x0, pos.y1+.025, f"({chr(ord('a')+index)}) {title}", fontsize=9.1,
                 weight="bold", ha="left", va="baseline")
        matrix.pcolormesh(np.arange(len(nodes_shown)+1)-.5, np.arange(len(y)+1)-.5,
                          selections, cmap=ListedColormap(FREQUENCIES), vmin=-.5, vmax=3.5,
                          edgecolors="white", linewidth=.5)
        matrix.set_yticks([])
        matrix.set_xticks(range(len(nodes_shown)), nodes_shown, rotation=60, ha="right",
                          rotation_mode="anchor")
        matrix.tick_params(axis="x", labelsize=5.6, length=2, pad=1)
        matrix.set_xlabel("Vertex ID", fontsize=6.6, labelpad=1)
        matrix.set_ylim(len(y)-.4, -.6)
        for axis in (bars, matrix):
            axis.axhline(len(y)-2.5, color=GRAY, linewidth=.55, linestyle="--")
        matrix.spines["left"].set_visible(False)
    box = dict(frameon=False, fontsize=6.5, borderaxespad=0, handlelength=1.05,
               handletextpad=.35, columnspacing=1.0, loc="lower center")
    fig.legend([Patch(facecolor=BLUE), Patch(facecolor=YELLOW)],
               ["Shared with Fused RLNN", "Other selected vertices"], ncol=2,
               bbox_to_anchor=(.5,.048), **box)
    keys = [Patch(facecolor=c, edgecolor=GRID, linewidth=.5) for c in FREQUENCIES]
    fig.legend(keys, ["0 of 3 seeds", "1 of 3", "2 of 3", "3 of 3"], ncol=4,
               bbox_to_anchor=(.5,.013), **box)
    fig.savefig(ROOT / "overleaf/figures/cross_scale_witnesses.pdf",
                metadata={"Title": "Delivered vertex sets across four graph scales", "Author": ""})
    plt.close(fig)


if __name__ == "__main__":
    data = load_cases()
    save_records(data)
    draw(data)
