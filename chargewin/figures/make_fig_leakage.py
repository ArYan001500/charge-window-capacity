"""Figure 2: inputs that carry the label."""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("CHARGEWIN_ROOT", ".")).resolve()
D = ROOT / "results/C83_FIGURE_DATA_2026-10-29"
cal = pd.read_csv(D / "calce_leak_predictions.csv")
soc = pd.read_csv(D / "soc_energy.csv")

HONEST, LEAKY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42, "axes.titlesize": 8})


def style(ax, title):
    ax.set_title(title, loc="left", color=INK, fontweight="bold")
    ax.grid(color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2)


fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.75), gridspec_kw={"wspace": 0.45})

ax = axes[0]
x = cal[cal.model == "window_only"]
ax.scatter(x.Q, x.dq, s=7, color=LEAKY, alpha=0.55, lw=0, zorder=3)
k = np.polyfit(x.Q, x.dq, 1)
gx = np.linspace(x.Q.min(), x.Q.max(), 10)
ax.plot(gx, np.polyval(k, gx), color=INK, lw=0.9, ls="--", zorder=4)
r = np.corrcoef(x.Q, x.dq)[0, 1]
ax.text(0.04, 0.93, f"r = {r:.3f}", transform=ax.transAxes, color=INK, va="top")
ax.set_xlabel("Reference capacity (Ah)")
ax.set_ylabel("Charge moved in the cycle (Ah)")
style(ax, "(a) Input tracks the label")

ax = axes[1]
lo, hi = cal.Q.min() - 0.02, cal.Q.max() + 0.02
ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls=":", zorder=2)
for model, color, lab in (("window_only", HONEST, "Voltage window only"), ("window_dq_dur", LEAKY, "+ cycle charge, duration")):
    m = cal[cal.model == model]
    mae = m.groupby("test_cell").apply(lambda z: (z.pred - z.Q).abs().mean()).mean()
    ax.scatter(m.Q, m.pred, s=7, color=color, alpha=0.55, lw=0, zorder=3, label=f"{lab}\nMAE {mae:.3f} Ah")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel("Reference capacity (Ah)")
ax.set_ylabel("Predicted capacity (Ah)")
ax.legend(frameon=False, fontsize=6.3, loc="lower right", handletextpad=0.2, borderaxespad=0.2, markerscale=1.5)
style(ax, "(b) Apparent accuracy")

ax = axes[2]
grp = pd.read_csv(ROOT / "data/uconn/cycling_parameters.csv", encoding="utf-8-sig").set_index("Cell ID")["Group Number"]
soc["group"] = soc.cell.map(grp)
lo, hi = soc.Q.min() - 0.03, soc.Q.max() + 0.03
ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls=":", zorder=2)
for col, color, lab in (("E_fixed", HONEST, "Fixed 0.9-Ah interval"), ("E_soc", LEAKY, "30–70% SOC interval")):
    pred = np.full(len(soc), np.nan)
    for g in soc.group.unique():
        tr, te = soc.group != g, soc.group == g
        k = np.polyfit(soc.loc[tr, col], soc.loc[tr, "Q"], 1)
        pred[te.values] = np.polyval(k, soc.loc[te, col])
    soc["p"] = pred
    mae = soc.groupby("group").apply(lambda z: (z.p - z.Q).abs().groupby(z.cell).mean().mean()).mean()
    ax.scatter(soc.Q, soc.p, s=6, color=color, alpha=0.5, lw=0, zorder=3 if col == "E_fixed" else 4, label=lab + "\n" + f"MAE {mae:.4f} Ah")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel("Reference capacity (Ah)")
ax.set_ylabel("Predicted capacity (Ah)")
ax.legend(frameon=False, fontsize=6.3, loc="lower right", handletextpad=0.2, borderaxespad=0.2, markerscale=1.5)
style(ax, "(c) SOC-defined segment")

out = Path(os.environ.get("CHARGEWIN_FIGURES", "figures")).resolve()
fig.savefig(out / "Figure_leakage.pdf", bbox_inches="tight")
fig.savefig(out / "Figure_leakage.png", dpi=300, bbox_inches="tight")
print("saved")
