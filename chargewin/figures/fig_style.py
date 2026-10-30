"""Shared style of the figures."""
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(os.environ.get("CHARGEWIN_ROOT", ".")).resolve()
RES = ROOT / "results"
OUT = Path(os.environ.get("CHARGEWIN_FIGURES", "figures")).resolve()

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#a8a69f", "#f3f2ee"

DATASETS = {
    "UConn": ("NMC/graphite", "natural start", 2.25),
    "HUST": ("LFP/graphite", "masked start", 1.1),
    "KIT": ("NMC/C–SiO", "semi-anchored start", 3.0),
}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42, "axes.titlesize": 8,
                     "axes.linewidth": 0.7, "legend.frameon": False})


def style(ax, title=None, grid="both"):
    if title:
        ax.set_title(title, loc="left", color=INK, fontweight="bold", pad=14)
    if grid:
        ax.grid(color=GRID, lw=0.6, zorder=0, axis=grid)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2)


def badge(ax, name, x=0.0, y=1.015):
    """Dataset badge drawn above the axes: bold name on a tinted pill, chemistry and start condition after it."""
    chem, start, _ = DATASETS[name]
    t = ax.text(x, y, name, transform=ax.transAxes, fontsize=7.0, fontweight="bold", color="white",
                va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.45", fc=INK, ec="none"))
    ax.annotate(f"{chem} · {start}", xy=(1, 0), xycoords=t, xytext=(5, 0), textcoords="offset points",
                fontsize=6.3, color=MUTED, va="bottom", ha="left")
    return t


def panel_letters(fig, axes_letters, dx=-0.045, dy=0.045):
    """Bold panel letters placed just above-left of the given axes (after layout is fixed)."""
    for ax, s in axes_letters:
        p = ax.get_position()
        fig.text(p.x0 + dx, p.y1 + dy, s, fontsize=8, fontweight="bold", color=INK, ha="left", va="bottom")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    print("saved", name)


__all__ = ["ROOT", "RES", "OUT", "BLUE", "ORANGE", "AQUA", "YELLOW", "VIOLET", "INK", "MUTED", "GRID", "BASE", "SURF",
           "DATASETS", "style", "badge", "save", "FancyBboxPatch", "plt"]
