"""Figure 6: fusion with sparse tests."""
import json

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from fig_style import (AQUA, BASE, BLUE, GRID, INK, MUTED, ORANGE, RES, VIOLET, plt, save, style)

c91 = pd.read_csv(RES / "C91_UCONN_CAUSAL_FUSION_2026-10-29_v1/per_group.csv")
c91 = c91[c91.variant == "causal_nested"]
c93 = pd.read_csv(RES / "C93_KIT_HUST_NESTED_FUSION_2026-10-29_v1/per_group.csv")
c32 = c93[c93.dataset == "KIT"]
own = pd.read_csv(RES / "C103_UCONN_OWN_RATE_2026-10-31_v1/per_group.csv")
ownk = pd.read_csv(RES / "C103b_KIT_HUST_OWN_RATE_2026-10-31_v1/per_group.csv")
ownk = ownk[ownk.dataset == "KIT"]
win_kit = c32[c32.k_every == 1].window.mean()

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), gridspec_kw={"wspace": 0.55, "width_ratios": [1.15, 1, 1]})

ax = axes[0]
T = np.array([7, 14, 28, 56])
series = [("Last test + training fade rate", c91.assign(mae=c91.ref_drift), BASE, "o", "-"),
          ("Last test + own fade rate", own.assign(mae=own.own_ref), VIOLET, "s", "-"),
          ("Scalar Kalman filter", c91.assign(mae=c91.kf_scalar), ORANGE, "^", "-"),
          ("Bias-aware filter, Eq. (4)", c91.assign(mae=c91.kf_bias), AQUA, "D", "-")]
wmean = c91.window.mean()
ax.axhline(wmean, color=BLUE, lw=1.3, ls=(0, (4, 2)), zorder=2, label="Window sensor alone")
for lab, d, color, mk, ls in series:
    g = d.groupby("T").mae
    m, q1, q3 = g.mean().reindex(T), g.quantile(0.25).reindex(T), g.quantile(0.75).reindex(T)
    ax.fill_between(T, q1, q3, color=color, alpha=0.14, lw=0, zorder=2)
    ax.plot(T, m, color=color, lw=1.8, marker=mk, ms=4.5, mec="white", mew=0.6, zorder=3, label=lab)
ref56 = own[own["T"] == 56].own_ref.mean()
kf56 = c91[c91["T"] == 56].kf_bias.mean()
ax.annotate("", xy=(56, kf56 + 0.002), xytext=(56, ref56 - 0.002),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, mutation_scale=7))
ax.text(56, kf56 - 0.006, f"−{(1 - kf56 / ref56):.0%}", ha="center", va="top", fontsize=6.4, color=INK,
        fontweight="bold")
ax.set_xscale("log", base=2)
ax.set_xticks(T)
ax.set_xticklabels([f"{t}" for t in T])
ax.set_xlim(6, 64)
ax.set_ylim(0, 0.15)
ax.set_xlabel("Minimum days between tests")
ax.set_ylabel("MAE on held-out groups (Ah)")
ax.legend(loc="upper left", fontsize=5.7, handlelength=1.6, borderaxespad=0.1)
style(ax, "(a) UConn: sparse tests")

ax = axes[1]
lim = (0, 0.3)
ax.plot(lim, lim, color=MUTED, lw=0.8, ls=":", zorder=1)
ax.fill_between(lim, [0, 0], lim, color="#e7f6f0", lw=0, zorder=0)
for k, mk, lab in ((1, "o", "test every check-up"), (2, "^", "every 2nd check-up")):
    d = c32[c32.k_every == k]
    o = ownk[ownk.k_every == k]
    ax.scatter(o.own_ref, o.kf_bias, s=24, marker=mk, color=AQUA, edgecolor="white", lw=0.6, zorder=3,
               label=f"{lab}: {(o.kf_bias < o.own_ref).sum()}/{len(o)}")
ax.axhline(win_kit, color=BLUE, lw=1.1, ls=(0, (4, 2)), zorder=2)
ax.text(0.006, win_kit - 0.004, "window sensor alone", color=BLUE, fontsize=6.0, ha="left",
        va="top")
ax.text(0.295, 0.004, "filter better", color=AQUA, fontsize=6.4, ha="right", va="bottom", fontweight="bold")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("Last check-up + own fade rate (Ah)")
ax.set_ylabel("Bias-aware filter (Ah)")
ax.legend(loc="upper left", fontsize=5.7, handletextpad=0.2, borderaxespad=0.1, title="KIT groups, filter better",
          title_fontsize=5.9)
style(ax, "(b) KIT: every held-out group")

ax = axes[2]
ax.axhspan(-0.06, 0, color="#fbe9e1", lw=0, zorder=0)
ax.axhline(0, color=INK, lw=0.8, zorder=1)
u = c91[c91["T"] == 14]
k = c32[c32.k_every == 1]
pts = pd.concat([pd.DataFrame({"ds": "UConn", "passive": u.window, "gain": u.window - u.kf_bias}),
                 pd.DataFrame({"ds": "KIT", "passive": k.window, "gain": k.window - k.bias_kf})])
for ds, color in (("UConn", BLUE), ("KIT", ORANGE)):
    d = pts[pts.ds == ds]
    ax.scatter(d.passive, d.gain, s=22, color=color, edgecolor="white", lw=0.6, zorder=3)
rho = pts[["passive", "gain"]].corr(method="spearman").iloc[0, 1]
ax.text(0.97, 0.3, f"Spearman ρ = {rho:.2f}\n{len(pts)} held-out groups", transform=ax.transAxes, ha="right",
        va="center", fontsize=6.0, color=INK)
ax.text(0.97, 0.05, "fusion hurts", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4, color=ORANGE,
        fontweight="bold")
ax.set_ylim(-0.06, 0.11)
ax.set_xlabel("Window-sensor error (Ah)")
ax.set_ylabel("Gain from fusion (Ah)")
h = [Line2D([], [], marker="o", ls="", color=BLUE, ms=4.5, label="UConn"),
     Line2D([], [], marker="o", ls="", color=ORANGE, ms=4.5, label="KIT")]
ax.legend(handles=h, loc="upper left", fontsize=5.7, handletextpad=0.1, borderaxespad=0.1)
style(ax, "(c) When fusion pays")
save(fig, "Figure_fusion")
print("rho", round(rho, 3), "ref56", round(ref56, 4), "kf56", round(kf56, 4))
