"""Figure 3: overview of the estimator."""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors

ROOT = Path(os.environ.get("CHARGEWIN_ROOT", ".")).resolve()
D = ROOT / "results/C94_OVERVIEW_CAUSAL_2026-10-29"
CELL = 21
curves = pd.read_csv(ROOT / "tables/C38_UCONN_REF_CURVES_2026-10-24_v1/uconn_ref_chg_QofV.csv.gz")
rpt = pd.read_csv(ROOT / "tables/C10_UCONN_2026-10-23_v1/uconn_rpt_table.csv.gz")[["cell", "wk", "Q_ref"]]
curves = curves[curves.cell == CELL].merge(rpt, on=["cell", "wk"]).dropna(subset=["Q_ref"]).sort_values("wk")
QC = [c for c in curves.columns if c.startswith("q")]
V = np.array([float(c[1:]) for c in QC])
track = pd.read_csv(D / f"overview_cell{CELL}_track.csv")
tests = pd.read_csv(D / f"overview_cell{CELL}_tests.csv")
iv = pd.read_csv(D / f"overview_cell{CELL}_intervals.csv")

BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42, "axes.titlesize": 8})


def style(ax, title):
    ax.set_title(title, loc="left", color=INK, fontweight="bold")
    ax.grid(color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2)


def curve(row):
    q = row[QC].values.astype(float)
    ok = ~np.isnan(q)
    qq = np.maximum.accumulate(q[ok])
    return qq - qq[0], V[ok]


fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.3), gridspec_kw={"hspace": 0.5, "wspace": 0.5})

ax = axes[0, 0]
norm = colors.Normalize(vmin=curves.Q_ref.min(), vmax=curves.Q_ref.max())
cmap = colors.LinearSegmentedColormap.from_list("age", ["#9ec5f0", "#2a78d6", "#0b3d7a"])
sel = curves.iloc[np.linspace(0, len(curves) - 1, 8).astype(int)]
for _, r in sel.iterrows():
    q, v = curve(r)
    ax.plot(q, v, color=cmap(1 - norm(r.Q_ref)), lw=1.1, zorder=3)
mid = sel.iloc[len(sel) // 2]
q, v = curve(mid)
for k, (a, lab) in enumerate(((0.15, "A"), (0.75, "B"), (1.25, "C"))):
    m = (q >= a) & (q <= a + 0.375)
    ax.plot(q[m], v[m], color=ORANGE, lw=3.2, solid_capstyle="round", zorder=4)
    ax.text(q[m][0], v[m][0] + 0.07, lab, color=ORANGE, fontweight="bold", ha="center", fontsize=7.5)
sm = cm.ScalarMappable(norm=norm, cmap=colors.LinearSegmentedColormap.from_list("age_r", ["#0b3d7a", "#2a78d6", "#9ec5f0"]))
cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
cb.set_label("Capacity (Ah)", fontsize=6.8)
cb.ax.tick_params(labelsize=6.3, length=2)
ax.set_xlabel("Charge since start of reference charge (Ah)")
ax.set_ylabel("Voltage (V)")
ax.text(0.98, 0.05, "segments A–C: equal charge,\nunknown start", transform=ax.transAxes, ha="right", va="bottom",
        fontsize=6.5, color=ORANGE)
style(ax, "(a) Charge curves age; segments start anywhere")

ax = axes[0, 1]
m = (q >= 0.75) & (q <= 0.75 + 0.375)
qw, vw = q[m] - q[m][0], v[m]
gq = np.linspace(0, qw[-1], 31)
gv = np.interp(gq, qw, vw)
ax.fill_between(gq, vw[0], gv, color=BLUE, alpha=0.18, lw=0, zorder=2, label="integral of V − V₁")
ax.plot(gq, gv, color=BLUE, lw=1.6, zorder=3)
ax.scatter(gq[::3], gv[::3], s=12, color=BLUE, zorder=4, label="voltage samples")
sm_v = np.convolve(gv, np.ones(5) / 5, mode="same")
sm_v[:2], sm_v[-2:] = gv[:2], gv[-2:]
d2 = np.gradient(np.gradient(sm_v, gq), gq)[3:-3]
i = int(np.argmax(np.abs(d2))) + 3
ax.scatter([gq[i]], [gv[i]], s=46, facecolor="white", edgecolor=ORANGE, lw=1.6, zorder=5, label="curvature extremum")
ax.axhline(vw[0], color=MUTED, lw=0.7, ls=":")
ax.set_xlabel("Charge within segment B (Ah)")
ax.set_ylabel("Voltage (V)")
ax.legend(frameon=False, fontsize=6.3, loc="lower right")
style(ax, "(b) Window descriptors (illustration)")

ax = axes[1, 0]
rp = track.dropna(subset=["Q"]).groupby("Q").day.min().reset_index().sort_values("day")
ax.plot(track.day, track.Q, color=INK, lw=1.2, zorder=3, label="reference capacity")
ax.scatter(track.day, track.pred, s=4, color=BLUE, alpha=0.45, lw=0, zorder=2,
           label=f"window estimate (MAE {np.nanmean(np.abs(track.pred - track.Q)[track.kf.notna()]):.3f} Ah)")
kt = track.dropna(subset=["kf"])
ax.plot(kt.day, kt.kf, color=AQUA, lw=1.5, zorder=4,
        label=f"bias-aware filter (MAE {np.nanmean(np.abs(track.kf - track.Q)):.3f} Ah)")
ax.scatter(tests.day, tests.test, marker="D", s=22, color=ORANGE, edgecolor="white", lw=0.6, zorder=5,
           label="pulse-based test (≥14 days apart)")
ax.set_xlabel("Time (days)")
ax.set_ylabel("Capacity (Ah)")
ax.legend(frameon=False, fontsize=6.2, loc="lower left")
style(ax, "(c) Bias-aware fusion with sparse tests")

ax = axes[1, 1]
iv = iv.sort_values("day")
ax.fill_between(iv.day, iv.lo, iv.hi, color=BLUE, alpha=0.14, lw=0, zorder=2, label="90% interval (other groups)")
ax.scatter(iv.day, iv.MW_quad, s=4, color=BLUE, alpha=0.5, lw=0, zorder=3, label="window estimate")
ax.plot(iv.day, iv.Q, color=INK, lw=1.2, zorder=4, label="reference capacity")
cov = ((iv.Q >= iv.lo) & (iv.Q <= iv.hi)).mean()
ax.text(0.02, 0.05, f"coverage in this cell: {cov:.0%}", transform=ax.transAxes, fontsize=6.6, color=INK)
ax.set_xlabel("Time (days)")
ax.set_ylabel("Capacity (Ah)")
ax.legend(frameon=False, fontsize=6.2, loc="upper right")
style(ax, "(d) 90% intervals of window estimates")

out = Path(os.environ.get("CHARGEWIN_FIGURES", "figures")).resolve()
fig.savefig(out / "Figure_overview.pdf", bbox_inches="tight")
fig.savefig(out / "Figure_overview.png", dpi=300, bbox_inches="tight")
print("saved")
