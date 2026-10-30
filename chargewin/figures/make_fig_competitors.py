"""Figure 5: adapted competitors."""
import json

import numpy as np
from matplotlib.lines import Line2D

from jes_style import (BASE, BLUE, GRID, INK, MUTED, ORANGE, RES, VIOLET, YELLOW, panel_letters, plt, save, style)

c78 = json.load(open(RES / "C78_TIM_COMPETITORS_2026-10-28_v1/summary.json"))
c79 = json.load(open(RES / "C79_EXACT_ARCH_2026-10-28_v1/summary.json"))
c55 = json.load(open(RES / "C55_KIT_TAOSEE_BASELINE_2026-10-24_v1/summary.json"))

rows, heads = [], []


def add(label, ours, comp, ci, color, marker, filled=True):
    rows.append((label, ours - comp, ci, color, marker, filled))


heads.append(("10-min windows, any start", len(rows)))
for ds, key in (("UConn", "G2_S1_UConn"), ("KIT", "G2_S1_KIT")):
    s = c78[key]
    add(f"{ds} · Guo calibration lookup", s["ours"], s["guo"], s["ours_minus_guo_CI95"], ORANGE, "o")
    add(f"{ds} · Shao indicator", s["ours"], s["shao_E"], s["ours_minus_shao_CI95"], VIOLET, "o")
    x = c79[f"shao_exact_{ds}"]
    add(f"{ds} · Shao BiLSTM-FM-Res", x["ours"], x["shao_exact"], x["ours_minus_shao_CI95"], VIOLET, "s")
heads.append(("Slow 30-min segments (UConn)", len(rows)))
for seg, name in (("LOW", "low"), ("MID", "mid-curve"), ("HIGH", "high")):
    s, x = c78["G1_slow_segments"][seg], c79["guo_exact_slow_segments"][seg]
    add(f"{name} · Guo calibration lookup", s["ours_F0F1"], s["guo"], s["ours_minus_guo_CI95"], ORANGE, "o")
    add(f"{name} · Guo IDATL-GRU", x["ours_F0F1"], x["guo_exact"], x["ours_minus_guo_CI95"], ORANGE, "s")
heads.append(("Post hoc: constant-current windows only", len(rows)))
add("UConn (11% of windows) · Guo lookup", 0.211, 0.109, [0.080, 0.121], ORANGE, "o", False)
add("KIT (99.6% of windows) · Guo lookup", 0.101, 0.205, [-0.118, -0.091], ORANGE, "o", False)

fig = plt.figure(figsize=(7.4, 4.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.32, 1], height_ratios=[1, 1], hspace=0.75, wspace=0.36)

ax = fig.add_subplot(gs[:, 0])
ypos, y = [], 0
starts = {i for _, i in heads}
for i in range(len(rows)):
    if i in starts:
        y += 0.9
    ypos.append(y)
    y += 1
ypos = np.array(ypos)
ax.axvspan(0, 0.13, color="#fbe9e1", zorder=0, lw=0)
ax.axvspan(-0.23, 0, color="#e6f0fb", zorder=0, lw=0)
ax.axvline(0, color=INK, lw=0.8, zorder=1)
for (label, d, ci, color, marker, filled), yy in zip(rows, ypos):
    ax.hlines(yy, ci[0], ci[1], color=color, lw=2.2, zorder=2)
    ax.scatter(d, yy, s=34, marker=marker, color=color if filled else "white", edgecolor=color, lw=1.2, zorder=3)
ax.set_yticks(ypos)
ax.set_yticklabels([r[0] for r in rows], fontsize=6.1)
for txt, i in heads:
    ax.text(-0.235, ypos[i] - 0.95, txt, fontsize=6.6, fontweight="bold", color=INK, ha="right", va="center",
            transform=ax.transData, clip_on=False)
ax.set_ylim(ypos[-1] + 1.6, -0.4)
ax.set_xlim(-0.23, 0.13)
ax.set_xlabel("MAE difference, ours − competitor (Ah)")
ax.text(-0.115, ypos[-1] + 1.45, "ours better", color=BLUE, fontsize=6.6, ha="center", va="bottom", fontweight="bold")
ax.text(0.065, ypos[-1] + 1.45, "competitor better", color=ORANGE, fontsize=6.6, ha="center", va="bottom",
        fontweight="bold")
style(ax, grid="x")
ax.tick_params(axis="y", length=0)
leg = [Line2D([], [], marker="o", ls="", color=MUTED, markersize=4.5, label="adapted indicator / rule"),
       Line2D([], [], marker="s", ls="", color=MUTED, markersize=4.5, label="rebuilt network"),
       Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED, markersize=4.5, label="post hoc")]
ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.32, -0.12), ncol=3, fontsize=5.9, handletextpad=0.1,
          columnspacing=0.8)

ax = fig.add_subplot(gs[0, 1])
methods = [("Ours (window sensor)", BLUE), ("Shao BiLSTM-FM-Res", VIOLET), ("Shao indicator", "#a99ee9"),
           ("Guo calibration lookup", ORANGE)]
vals = {"UConn": [c78["G2_S1_UConn"]["ours"], c79["shao_exact_UConn"]["shao_exact"], c78["G2_S1_UConn"]["shao_E"],
                  c78["G2_S1_UConn"]["guo"]],
        "KIT": [c78["G2_S1_KIT"]["ours"], c79["shao_exact_KIT"]["shao_exact"], c78["G2_S1_KIT"]["shao_E"],
                c78["G2_S1_KIT"]["guo"]]}
h = 0.19
for k, ds in enumerate(vals):
    for m, ((name, color), v) in enumerate(zip(methods, vals[ds])):
        yy = k + (m - 1.5) * (h + 0.02)
        ax.barh(yy, v, height=h, color=color, zorder=2, lw=0)
        short = ("ours", "Shao network", "Shao indicator", "Guo lookup")[m]
        ax.text(v + 0.004, yy, f"{v:.3f}  {short}" if k == 0 else f"{v:.3f}", va="center", fontsize=5.8, color=INK)
ax.set_yticks([0, 1])
ax.set_yticklabels(["UConn", "KIT"], fontweight="bold")
ax.set_ylim(1.55, -0.55)
ax.set_xlim(0, 0.5)
ax.set_xlabel("MAE on held-out groups (Ah)")
style(ax, "(b) Ordinary 10-min windows", grid="x")
ax.tick_params(axis="y", length=0)

ax = fig.add_subplot(gs[1, 1])
c85 = json.load(open(RES / "C85_POSTHOC_CI_2026-10-29_v1/summary.json"))["A_taosee"]
c87 = json.load(open(RES / "C87_TAOSEE_UCONN_HUST_2026-10-29_v1/summary.json"))
k = "3.60-3.70V"
cases = [("KIT", k, c85[k]["reproduced"]["TS_mlp"], c85[k]["reproduced"]["MW_quadT"], c55[k]["coverage"],
          c85[k]["ours_minus_TS_mlp"]["CI95"], "development")]
for ds, key in (("UConn", "UConn"), ("HUST masked", "HUST_masked")):
    w = c87[key]["primary"]
    r = c87[key][w]
    cases.append((ds, w, r["TS_mlp"], r["MW_quad"], r["coverage"], r["ours_minus_TS_mlp"]["CI95"], "pre-registered"))
x = np.arange(len(cases))
for i, (ds, w, ts, ours, cov, ci, status) in enumerate(cases):
    ax.bar(i - 0.19, ts, width=0.36, color=YELLOW, zorder=2, lw=0, label="Tao & See Q–V window (MLP)" if i == 0 else None)
    ax.bar(i + 0.19, ours, width=0.36, color=BLUE, zorder=2, lw=0, label="Ours, same windows" if i == 0 else None)
    verdict = "ours better" if ci[1] < 0 else ("theirs better" if ci[0] > 0 else "n.s.")
    ax.text(i, max(ts, ours) + 0.012, verdict + "\n" + f"{cov:.0%} of windows", ha="center", va="bottom",
            fontsize=5.6, color=INK if verdict != "n.s." else MUTED, linespacing=1.05)
ax.set_xticks(x)
ax.set_xticklabels([c[0] + "\n" + c[1].replace("-", "–").replace("V", " V") + "\n" + f"({c[6]})" for c in cases],
                   fontsize=5.6)
ax.set_ylabel("MAE (Ah)")
ax.set_ylim(0, 0.34)
ax.set_yticks(np.arange(0, 0.31, 0.05))
style(ax, "(c) Fixed voltage window (Tao & See)", grid="y")
ax.legend(loc="upper right", fontsize=5.6, handlelength=0.9, handletextpad=0.4, borderaxespad=0.0, ncol=2,
          bbox_to_anchor=(1.0, 1.02))

fig.canvas.draw()
pa = fig.axes[0].get_position()
fig.text(0.0, pa.y1 + 0.035, "(a) Guo and Shao: paired difference with 95% CI", fontsize=8, fontweight="bold",
         color=INK)
save(fig, "Figure_competitors")
