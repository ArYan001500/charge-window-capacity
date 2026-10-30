"""Figure 1: datasets."""
import json

import numpy as np
import pandas as pd
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from fig_style import BASE, BLUE, INK, MUTED, ORANGE, RES, badge, plt, save, style

D = RES / "C88_DATASET_FIGURE_DATA_2026-10-29"
cur = pd.read_csv(D / "example_curves.csv.gz")
traj = pd.read_csv(D / "capacity_trajectories.csv.gz", dtype={"cell": str})
sm = json.load(open(D / "summary.json"))
cmap = colors.LinearSegmentedColormap.from_list("cap", ["#f2b38f", "#9ec5f0", "#2a78d6", "#0b3d7a"])
norm = colors.Normalize(0.6, 1.0)
AGE = {"UConn": "Time (days)", "HUST": "Cycle", "KIT": "Time (days)"}
EX = {"UConn": sm["UConn_cell"], "HUST": sm["HUST_cell"], "KIT": sm["KIT_cell"]}

fig = plt.figure(figsize=(7.4, 5.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 0.8], hspace=0.3, wspace=0.08)
for j, ds in enumerate(("UConn", "HUST", "KIT")):
    c = cur[cur.dataset == ds]
    shape = c.groupby("age").v.apply(lambda v: np.interp(np.linspace(0, 1, 50), np.linspace(0, 1, len(v)), v.values))
    S = np.vstack(shape.values)
    dev = np.abs(S - np.median(S, axis=0)).mean(1)
    mad = np.median(np.abs(dev - np.median(dev)))
    c = c[c.age.isin(shape.index[dev <= np.median(dev) + 5 * mad])]
    ages = np.sort(c.age.unique())
    q0 = c[c.age == ages[0]].Q.iloc[0]
    ax = fig.add_subplot(gs[0, j], projection="3d")
    zmin = c.v.min() - 0.05
    for n, a in enumerate(ages):
        z = c[c.age == a]
        col = cmap(norm(z.Q.iloc[0] / q0))
        if n % 4 == 0:
            verts = [list(zip(z.q, [a] * len(z), z.v)) + [(z.q.iloc[-1], a, zmin), (z.q.iloc[0], a, zmin)]]
            ax.add_collection3d(Poly3DCollection(verts, facecolor=col, alpha=0.05, edgecolor="none"))
        ax.plot(z.q, [a] * len(z), z.v, color=col, lw=0.9)
    mid = c[c.age == ages[len(ages) // 2]]
    span = sm[f"median_Q600_{ds}"]
    for f in (0.08, 0.45, 0.8):
        q1 = f * (mid.q.max() - span)
        m = (mid.q >= q1) & (mid.q <= q1 + span)
        ax.plot(mid.q[m], [ages[len(ages) // 2]] * int(m.sum()), mid.v[m], color=ORANGE, lw=2.8, solid_capstyle="round")
    ax.set_xlabel("Charge (Ah)", fontsize=6.4, labelpad=-6)
    ax.set_ylabel(AGE[ds], fontsize=6.4, labelpad=-5)
    ax.set_zlabel("Voltage (V)", fontsize=6.4, labelpad=-6)
    ax.set_zlim(zmin, c.v.max() + 0.02)
    ax.tick_params(labelsize=5.4, pad=-3)
    ax.view_init(elev=24, azim=-58)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("#f7f6f3")
        axis.pane.set_edgecolor("#e4e3df")
        axis._axinfo["grid"].update(color="#e4e3df", linewidth=0.5)
    ax.set_box_aspect((1.25, 1.25, 0.8))
    ax.text2D(0.02, 0.98, f" {ds} ", transform=ax.transAxes, fontsize=7.0, fontweight="bold", color="white",
              va="top", bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.45", fc=INK, ec="none"))
    ax.text2D(0.2 + 0.035 * len(ds), 0.975, {"UConn": "NMC/graphite · C/3",
                                             "HUST": "LFP/graphite",
                                             "KIT": "NMC/C–SiO"}[ds],
              transform=ax.transAxes, fontsize=6.0, color=MUTED, va="top")

    ax2 = fig.add_subplot(gs[1, j])
    t = traj[traj.dataset == ds]
    for cell, z in t.groupby("cell"):
        if cell != EX[ds]:
            ax2.plot(z.age, z.Q, color=BASE, lw=0.45, alpha=0.55)
    z = t[t.cell == EX[ds]]
    ax2.plot(z.age, z.Q, color=BLUE, lw=1.6)
    ng = {"UConn": 11, "HUST": 7, "KIT": 13}[ds]
    ax2.text(0.97, 0.95, f"{t.cell.nunique()} cells, {ng} usage groups", transform=ax2.transAxes, ha="right",
             va="top", fontsize=6.2, color=INK)
    ax2.set_xlabel(AGE[ds])
    if j == 0:
        ax2.set_ylabel("Capacity (Ah)")
    style(ax2)

cax = fig.add_axes([0.955, 0.56, 0.012, 0.3])
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_label("Capacity / first curve", fontsize=6.3)
cb.ax.tick_params(labelsize=5.8, length=1.5)
cb.outline.set_visible(False)
fig.text(0.07, 0.905, "(a) Charge curves of one cell over its life; orange: charge span of a median scored 10-min window",
         fontsize=8, fontweight="bold", color=INK)
fig.text(0.07, 0.392, "(b) Capacity of every cell (example cell in blue)", fontsize=8, fontweight="bold", color=INK)
save(fig, "Figure_datasets")
print({ds: int(cur[cur.dataset == ds].age.nunique()) for ds in ("UConn", "HUST", "KIT")})
