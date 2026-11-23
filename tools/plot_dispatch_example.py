"""Draw the four-shipment worked example from its six candidate intervals."""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from paper_figure_style import BLUE, CREAM, GREEN, GRID, INK, apply_style, panel_title

TRIPS = {"a1": (0, 35), "a2": (10, 45), "b1": (25, 65),
         "b2": (35, 65), "c2": (50, 75), "d1": (70, 100)}
SELECTED = {"a2", "b1", "c2", "d1"}


def clock(minutes):
    return f"{8 + minutes // 60:02d}:{minutes % 60:02d}"


def label(trip):
    return f"${trip[0]}_{trip[1]}$"


def graph_edges():
    edges = {}
    for a, b in combinations(TRIPS, 2):
        lo_a, hi_a = TRIPS[a]
        lo_b, hi_b = TRIPS[b]
        if a[0] == b[0]:
            edges[a, b] = "shipment"
        elif a[1] == b[1] and max(lo_a, lo_b) < min(hi_a, hi_b):
            edges[a, b] = "vehicle"
    if len(edges) != 5 or any(set(pair) <= SELECTED for pair in edges):
        raise ValueError("The example must encode five incompatibilities and a feasible four-trip plan")
    return edges


def timetable(ax):
    rows = {"a1": 5.28, "b1": 4.50, "d1": 3.72, "a2": 2.15, "b2": 1.37, "c2": .59}
    ax.set(xlim=(-10, 102), ylim=(0, 6.35), xticks=[0, 25, 50, 75, 100], yticks=[])
    ax.set_xticklabels([clock(x) for x in ax.get_xticks()])
    for vehicle, lower, fill in ((1, 3.17, "#ECF1F8"), (2, .04, CREAM)):
        ax.add_patch(Rectangle((-10, lower), 112, 3.08, facecolor=fill, edgecolor=GRID, lw=.55, zorder=0))
        ax.text(-9, lower + 2.71, f"Vehicle {vehicle}", va="center", fontsize=7.2, weight="bold")
    for tick in ax.get_xticks():
        for lower in (.12, 3.25):
            ax.plot([tick, tick], [lower, lower + 2.33], color=GRID, lw=.55, ls=":", zorder=.5)
    for trip, (lo, hi) in TRIPS.items():
        selected = trip in SELECTED
        y = rows[trip]
        ax.text(-2.1, y, label(trip), ha="right", va="center", fontsize=8.2)
        ax.add_patch(Rectangle((lo, y - .28), hi - lo, .56, facecolor=GREEN if selected else "white",
                               edgecolor=INK, lw=.65, linestyle="-" if selected else (0, (3, 1.5))))
        ax.text((lo + hi) / 2, y, f"{clock(lo)}–{clock(hi)}", ha="center", va="center", fontsize=6.2)
    for edge in ("left", "right", "top"):
        ax.spines[edge].set_visible(False)
    ax.set_xlabel("Depot time", labelpad=2.5)


def conflict_graph(ax):
    positions = {"a1": (0, 1), "b1": (1, 1), "d1": (2, 1),
                 "a2": (0, 0), "b2": (1, 0), "c2": (2, 0)}
    styles = {"shipment": (INK, (0, (3, 2))), "vehicle": (BLUE, "-")}
    for vehicle, y, fill in ((1, 1, "#ECF1F8"), (2, 0, CREAM)):
        ax.add_patch(Rectangle((-.37, y - .44), 2.74, .88, facecolor=fill, edgecolor=GRID, lw=.55, zorder=0))
        ax.text(2.29, y + .31, f"Vehicle {vehicle}", ha="right", va="center", fontsize=6.8, weight="bold")
    for (a, b), kind in graph_edges().items():
        color, style = styles[kind]
        ax.plot(*zip(positions[a], positions[b]), lw=1.3, ls=style, color=color, zorder=1)
    for trip, (x, y) in positions.items():
        selected = trip in SELECTED
        ax.scatter(x, y, s=280 if selected else 235, marker="o" if selected else "s",
                   c=GREEN if selected else "white", edgecolors=INK, lw=.8, zorder=2)
        ax.text(x, y, label(trip), ha="center", va="center", fontsize=9, zorder=3)
    ax.set(xlim=(-.38, 2.38), ylim=(-.46, 1.48))
    ax.set_axis_off()


def main():
    apply_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.75, 2.73),
                                      gridspec_kw={"width_ratios": [1.52, 1]})
    fig.subplots_adjust(left=.017, right=.99, top=.866, bottom=.30, wspace=.12)
    panel_title(left, "a", "Six candidate trips")
    panel_title(right, "b", "Four compatible assignments")
    timetable(left)
    conflict_graph(right)
    handles = [Line2D([], [], color=BLUE, lw=1.3, label="Overlapping vehicle use"),
               Line2D([], [], color=INK, lw=1.3, ls=(0, (3, 2)), label="Same shipment"),
               Line2D([], [], marker="o", mfc=GREEN, mec=INK, mew=.65, ms=5, lw=0,
                      label="Selected trip"),
               Line2D([], [], marker="s", mfc="white", mec=INK, mew=.65, ms=4.5, lw=0,
                      label="Other trip")]
    legend = fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(.5, .025),
                        handlelength=1.55, columnspacing=1.2, fontsize=6.6)
    legend.get_frame().set_linewidth(.55)
    dest = Path(__file__).resolve().parents[1] / "overleaf/figures/dispatch_example.pdf"
    fig.savefig(dest,metadata={"Title":"Worked example in cold-chain dispatch", "Author":""})
    plt.close(fig)


if __name__ == "__main__":
    main()
