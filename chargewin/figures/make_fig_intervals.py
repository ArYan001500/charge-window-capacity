"""Figure 7: intervals under usage shift."""
import json

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from fig_style import (AQUA, BASE, BLUE, GRID, INK, MUTED, ORANGE, RES, badge, plt, save, style)

_c92 = pd.read_csv(RES / "C92_NESTED_RULES_AD_2026-10-29_v1/per_group.csv")
_c92 = _c92[_c92.contract == "nested"].copy()
KEY = {"A": "A_pooled", "B": "B_groupmax", "C": "C_norm", "D": "D_norm_groupmax"}
_c92["method"] = _c92.rule.map(KEY)
pg = _c92
_s = json.load(open(RES / "C92_NESTED_RULES_AD_2026-10-29_v1/summary.json"))["nested"]
sm = {KEY[r]: v for r, v in _s.items()}
qmean = {"UConn": pd.read_csv(RES / "C11_UCONN_LOGO_2026-10-23_v1/scored_logo.csv.gz", usecols=["Q"]).Q.mean(),
         "HUST": pd.read_csv(RES / "C14_HUST_LOGO_2026-10-23_v1/scored_random.csv.gz", usecols=["Q"]).Q.mean(),
         "KIT": pd.read_csv(RES / "C31_KIT_LOGO_2026-10-24_usage/scored.csv.gz", usecols=["Q"]).Q.mean()}
for ds in qmean:
    for m in ("A_pooled", "D_norm_groupmax"):
        x = pg[(pg.dataset == ds) & (pg.method == m)]
        assert abs(x.coverage.min() - sm[m.replace("A_pooled", "A_pooled")][ds]["min_cov"]) < 6e-4

fig = plt.figure(figsize=(7.4, 2.85))
gs = fig.add_gridspec(1, 4, width_ratios=[1.1, 0.8, 1.25, 1.35], wspace=0.32)

for j, ds in enumerate(("UConn", "HUST", "KIT")):
    ax = fig.add_subplot(gs[0, j])
    a = pg[(pg.dataset == ds) & (pg.method == "A_pooled")].set_index("group").coverage
    d = pg[(pg.dataset == ds) & (pg.method == "D_norm_groupmax")].set_index("group").coverage
    order = a.sort_values().index
    x = np.arange(len(order))
    ax.axhspan(0.2, 0.8, color="#fbe9e1", lw=0, zorder=0)
    ax.axhline(0.9, color=INK, lw=0.8, ls=(0, (4, 2)), zorder=1)
    for i, g in enumerate(order):
        ax.plot([i, i], [a[g], d[g]], color="#b9d3f3", lw=1.6, zorder=2)
    ax.scatter(x, a[order], s=20, facecolor="white", edgecolor=BASE, lw=1.1, zorder=3)
    ax.scatter(x, d[order], s=22, color=BLUE, edgecolor="white", lw=0.5, zorder=4)
    ax.set_xticks([])
    ax.set_xlim(-0.8, len(order) - 0.2)
    ax.set_ylim(0.2, 1.03)
    ax.set_xlabel(f"{len(order)} held-out groups", fontsize=6.4)
    if j == 0:
        ax.set_ylabel("Coverage of 90% interval")
        ax.text(0.04, 0.03, "below 0.80", transform=ax.transAxes, fontsize=6.0, color=ORANGE, fontweight="bold")
    else:
        ax.set_yticklabels([])
    ax.text(len(order) - 0.3, 0.885, "nominal", fontsize=5.8, color=MUTED, ha="right", va="top")
    ax.text(0.5, 0.17, f"worst group\n{a.min():.3f} → {d.min():.3f}", transform=ax.transAxes, ha="center",
            fontsize=6.2, color=INK, fontweight="bold", linespacing=1.2)
    style(ax, grid="y")
    badge(ax, ds)
    ax.texts[-1].set_text("")
    if j == 0:
        first = ax
h = [Line2D([], [], marker="o", ls="", mfc="white", mec=BASE, ms=4.5, label="pooled quantile (rule A)"),
     Line2D([], [], marker="o", ls="", color=BLUE, ms=4.5, label="group-max, Eq. (5), strictly nested")]
fig.legend(handles=h, loc="lower left", bbox_to_anchor=(first.get_position().x0 - 0.01, -0.06), ncol=2, fontsize=6.3,
           handletextpad=0.2)

ax = fig.add_subplot(gs[0, 3])
meth = [("A_pooled", "A pooled", BASE), ("C_norm", "C normalized", AQUA), ("B_groupmax", "B group-max", ORANGE),
        ("D_norm_groupmax", "D = Eq. (5)", BLUE)]
mk = {"UConn": "o", "HUST": "^", "KIT": "s"}
ax.axhspan(0.2, 0.8, color="#fbe9e1", lw=0, zorder=0)
for key, lab, color in meth:
    for ds in mk:
        s = sm[key][ds]
        ax.scatter(100 * s["mean_width_Ah"] / qmean[ds], s["min_cov"], s=30, marker=mk[ds], color=color,
                   edgecolor="white", lw=0.6, zorder=3)
        if key == "D_norm_groupmax":
            ax.annotate(ds, (100 * s["mean_width_Ah"] / qmean[ds], s["min_cov"]),
                        xytext={"UConn": (4, -6), "HUST": (4, 0), "KIT": (-4, 7)}[ds],
                        textcoords="offset points", fontsize=6.0, color=INK, va="center")
ax.axhline(0.8, color=ORANGE, lw=0.8, ls=":", zorder=1)
ax.set_ylim(0.2, 1.03)
ax.set_xlim(10, 40)
ax.set_xlabel("Mean interval width (% of mean capacity)")
ax.set_ylabel("Worst-group coverage")
style(ax, "(b) Price of guarding the worst group")
hm = [Line2D([], [], marker="o", ls="", color=c, ms=4.5, label=l) for _, l, c in meth]
hd = [Line2D([], [], marker=m, ls="", color=MUTED, ms=4.0, label=d) for d, m in mk.items()]
ax.legend(handles=hm + hd, loc="lower right", fontsize=5.6, handletextpad=0.1, borderaxespad=0.2, labelspacing=0.3,
          ncol=2, columnspacing=0.6)

fig.canvas.draw()
p = first.get_position()
fig.text(p.x0 - 0.045, p.y1 + 0.09, "(a) Coverage of every held-out usage group", fontsize=8, fontweight="bold",
         color=INK)
save(fig, "Figure_intervals")
print({ds: round(v, 4) for ds, v in qmean.items()})
