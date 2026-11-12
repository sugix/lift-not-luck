"""Shared chart style for the book: thin marks, recessive axes, fixed colours.

Categorical colours are assigned in a fixed order and follow the entity,
never its rank, so "Thompson" is the same colour in every chart.
"""

import matplotlib.pyplot as plt

# Validated categorical palette, slots 1 to 4 (blue, orange, aqua, yellow).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#1c1e26"
MUTED = "#5d5f65"
GRID = "#e6e9ee"

STRATEGY_COLOURS = {
    "A/B test (fixed split)": SERIES[0],
    "Epsilon-greedy": SERIES[1],
    "UCB1": SERIES[2],
    "Thompson sampling": SERIES[3],
}


def apply():
    plt.rcParams.update({
        "figure.figsize": (7.5, 4.2),
        "figure.dpi": 110,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "font.size": 10,
    })


def label_line_end(ax, x, y, text, colour):
    """Direct label at the right end of a line: colour marker, ink text."""
    ax.plot([x], [y], "o", color=colour, markersize=6, clip_on=False)
    ax.annotate(text, xy=(x, y), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK, annotation_clip=False)
