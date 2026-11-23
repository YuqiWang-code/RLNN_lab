"""Shared print styling for the MIS manuscript's vector figures."""

import matplotlib as mpl


BLUE = "#355B9B"
GREEN = "#78A64D"
ORANGE = "#DF842F"
YELLOW = "#F6D977"
GRAY = "#979796"
SKY = "#82AFD1"
INK = "#242525"
CREAM = "#F8F2E9"
GRID = "#CEC7BC"
GOLD_LINE = "#997318"
MUTED = "#727577"


def apply_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 7.6, "pdf.fonttype": 42,
        "mathtext.fontset": "dejavuserif", "axes.labelsize": 7.7,
        "axes.titlesize": 9.0, "axes.titleweight": "bold", "axes.titlepad": 6,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
        "axes.linewidth": .65, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.labelsize": 7.1, "ytick.labelsize": 7.1,
        "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "legend.fontsize": 6.9, "legend.frameon": True, "legend.fancybox": False,
        "legend.framealpha": 1, "legend.facecolor": "white", "legend.edgecolor": "#99958C",
        "legend.borderpad": .35, "legend.handletextpad": .45, "legend.columnspacing": 1.1,
        "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": .60,
    })


def style_axis(ax, grid="y", cream=True):
    ax.set_facecolor(CREAM if cream else "white")
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=.6, linestyle="-" if cream else ":")


def panel_title(ax, letter, title):
    ax.set_title(f"({letter.lower()}) {title}", loc="left")


def boxed_legend(ax, **kwargs):
    legend = ax.legend(**kwargs)
    legend.get_frame().set_linewidth(.55)
    return legend
