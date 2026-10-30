"""Figure 4: transfer to held-out usage groups."""
import json

import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from jes_style import (AQUA, BASE, BLUE, DATASETS, GRID, INK, MUTED, RES, badge, panel_letters, plt, save, style)

SRC = {
    "UConn": ("C11_UCONN_LOGO_2026-10-23_v1/scored_logo.csv.gz", "MW_quad", "C11_UCONN_LOGO_2026-10-23_v1", None),
    "HUST": ("C14_HUST_LOGO_2026-10-23_v1/scored_random.csv.gz", "MW_quad", "C14_HUST_LOGO_2026-10-23_v1", "random"),
    "KIT": ("C31_KIT_LOGO_2026-10-24_usage/scored.csv.gz", "MW_quadT", "C31_KIT_LOGO_2026-10-24_usage", None),
}


def eqcell(d, col):
    """Equal-cell MAE per group: windows -> cell mean -> group mean."""
    e = (d[col] - d.Q).abs().groupby([d.group, d.cell]).mean()
    return e.groupby(level=0).mean()


data = {}
for name, (f, wcol, cap, sub) in SRC.items():
    d = pd.read_csv(RES / f, usecols=lambda c: c in {"cell", "group", "Q", "M0", "P2", wcol})
    g = pd.DataFrame({m: eqcell(d, m) for m in ("M0", wcol, "P2")}).rename(columns={wcol: "W"})
    s = json.load(open(RES / cap / "summary.json"))
    s = (s[sub] if sub else s)["primary_mean_over_groups"]
    for m, k in (("M0", "M0"), ("W", wcol), ("P2", "P2")):
        assert abs(g[m].mean() - s[k]) < 6e-5, (name, m, g[m].mean(), s[k])
    data[name] = (d.rename(columns={wcol: "W"}), g)
    print(name, g.mean().round(4).to_dict(), "W beats M0 in", int((g.W < g.M0).sum()), "of", len(g))

cmap = colors.LinearSegmentedColormap.from_list("dens", ["#e3eefb", "#7fb0ea", "#2a78d6", "#0b3d7a"])
fig = plt.figure(figsize=(7.4, 6.0))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.5, wspace=0.42)


for j, name in enumerate(SRC):
    d, g = data[name]
    _, _, nom = DATASETS[name]
    ax = fig.add_subplot(gs[0, j])
    lo, hi = np.percentile(d.Q, 0.2) - 0.05, np.percentile(d.Q, 99.8) + 0.05
    hb = ax.hexbin(d.Q, d.W.clip(lo, hi), gridsize=38, extent=(lo, hi, lo, hi), cmap=cmap, bins="log", mincnt=1,
                   lw=0.15, edgecolors="white", zorder=2)
    ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8, ls=(0, (4, 2)), zorder=3)
    ax.axhline(d.M0.mean(), color=BASE, lw=1.4, zorder=3)
    ax.text(hi - 0.02 * (hi - lo), d.M0.mean() - 0.02 * (hi - lo), "training mean", color=MUTED, fontsize=6.0,
            ha="right", va="top", bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85), zorder=4)
    mae_w, mae_0 = g.W.mean(), g.M0.mean()
    rel = mae_w / nom if nom else mae_w / d.Q.mean()
    ax.text(0.04, 0.96, f"window sensor  {mae_w:.3f} Ah\n({rel:.1%} of {'nominal' if nom else 'mean'} capacity)",
            transform=ax.transAxes, va="top", fontsize=6.4, color=INK, linespacing=1.3, zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Reference capacity (Ah)")
    if j == 0:
        ax.set_ylabel("Estimated capacity (Ah)")
    style(ax)
    badge(ax, name)
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=5.8, length=1.5)
    cb.outline.set_visible(False)
    if j == 2:
        cb.set_label("Windows per bin", fontsize=6.2)

    ax = fig.add_subplot(gs[1, j])
    g = g.sort_values("W", ascending=True).reset_index()
    lab = g.group + (0 if name == "UConn" else 1)
    y = np.arange(len(g))
    ax.hlines(y, g.W, g.M0, color="#b9d3f3", lw=2.6, zorder=2)
    ax.scatter(g.M0, y, s=26, color=BASE, edgecolor="white", lw=0.6, zorder=3)
    ax.scatter(g.W, y, s=30, color=BLUE, edgecolor="white", lw=0.6, zorder=4)
    ax.scatter(g.P2, y, s=26, marker="D", color=AQUA, edgecolor="white", lw=0.6, zorder=4)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(0.0012 if name == "HUST" else 0.006, 6)
    ax.set_yticks(y)
    ax.set_yticklabels([f"G{v}" for v in lab], fontsize=5.8)
    ax.set_ylim(-0.7, len(g) - 0.3)
    ax.set_xlabel("Group MAE (Ah, log scale)")
    wins = int((g.W < g.M0).sum())
    ax.text(0.98, 0.5, f"window\nbeats mean\n{wins}/{len(g)}", transform=ax.transAxes, ha="right", va="center",
            fontsize=6.3, color=INK, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.6))
    style(ax, grid="x")
    badge(ax, name)
    if j == 0:
        ax.set_ylabel("Held-out usage group")

handles = [Line2D([], [], marker="o", ls="", color=BASE, markersize=5, label="Training-set mean (no measurement)"),
           Line2D([], [], marker="o", ls="", color=BLUE, markersize=5, label="Window sensor, 10-min window only"),
           Line2D([], [], marker="D", ls="", color=AQUA, markersize=4.5,
                  label="Window + capacity at last reference test (Eq. 2)")]
fig.canvas.draw()
r0, r1 = fig.axes[0].get_position(), fig.axes[-1].get_position()
fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, r1.y1 + 0.035), fontsize=6.6,
           handletextpad=0.3, columnspacing=1.6)
fig.text(r0.x0 - 0.06, r0.y1 + 0.045, "(a) Every held-out window", fontsize=8, fontweight="bold", color=INK)
fig.text(r0.x0 - 0.06, r1.y1 + 0.075, "(b) Every held-out usage group", fontsize=8, fontweight="bold", color=INK)
save(fig, "Figure_transfer")
