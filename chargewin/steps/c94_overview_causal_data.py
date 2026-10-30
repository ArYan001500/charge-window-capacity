"""Data for the overview figure."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, CELL = Path(sys.argv[1]), int(sys.argv[2])
OUT = ROOT / "results/C94_OVERVIEW_CAUSAL_2026-10-29"
OUT.mkdir(parents=True, exist_ok=True)
T = 14
src91 = (Path(__file__).with_name("c91_uconn_causal_nested_fusion.py")).read_text(encoding="utf-8")
ns = {"__name__": "c94"}
sys.argv = ["c91", str(ROOT), str(OUT / "_unused")]
exec(compile(src91[:src91.index("nest = {m: {} for m in MODELS}")].replace("OUT.mkdir(parents=True, exist_ok=False)", ""), "c91_head", "exec"), ns)
d, g, y, groups, MODELS = ns["d"], ns["g"], ns["y"], ns["groups"], ns["MODELS"]
k = int(d.loc[d.cell == CELL, "group"].iloc[0])
nest = {}
for m in ("W", "WR"):
    S = MODELS[m]
    p = np.full(len(d), np.nan)
    for h in groups:
        if h != k:
            p[g == h] = S.predict([j for j in groups if j not in (k, h)], S.X[g == h])
    nest[m] = p
day = ((d.t - d.t.min()).dt.total_seconds() / 86400).values
ns.update(day=day, tday=((d.t_next - d.t.min()).dt.total_seconds() / 86400).values,
          cell_rows={c: np.where(d.cell.values == c)[0] for c in np.unique(d.cell)},
          cells_by_group={kk: list(np.unique(d.cell[g == kk])) for kk in groups})
ns["slope"] = {c: np.polyfit(day[ix], y[ix], 1)[0] for c, ix in ns["cell_rows"].items()}
exec(compile(src91[src91.index("def events("):src91.index("def evaluate(")], "c91_funcs", "exec"), ns)
run, events, fit_params, group_mae = ns["run"], ns["events"], ns["fit_params"], ns["group_mae"]
cbg = ns["cells_by_group"]
src_cells = [c for kk, cs in cbg.items() if kk != k for c in cs]
pr = fit_params(src_cells, nest["W"], nest["WR"])
grid = [(qn, qb) for qn in (1e-5, 1e-4, 1e-3) for qb in (1e-6, 1e-5, 1e-4, 1e-3)]
best = min(grid, key=lambda q: np.mean([group_mae(cbg[kk], {c: run(c, nest["W"], nest["WR"], T, "bias", pr["drift"], q[0], q[1],
                                                                   pr["rt"], pr["rw_in"], pr["pb0"]) for c in cbg[kk]})
                                        for kk in groups if kk != k]))
est = run(CELL, d.L_W.values, d.L_WR.values, T, "bias", pr["drift"], best[0], best[1], pr["rt"], pr["rw_in"], pr["pb0"])
ev, _ = events(CELL, d.L_W.values, d.L_WR.values, T)
x = d[d.cell == CELL].copy()
x["day"] = day[x.index]
x["kf"] = x.index.map(est)
x["pred"] = x.L_W
pd.DataFrame([(t, v) for t, kind, v in ev if kind == 0], columns=["day", "test"]).to_csv(OUT / f"overview_cell{CELL}_tests.csv", index=False)
x[["day", "Q", "pred", "kf"]].to_csv(OUT / f"overview_cell{CELL}_track.csv", index=False)

src92 = (Path(__file__).with_name("c92_nested_interval_rules.py")).read_text(encoding="utf-8")
n2 = {"__name__": "c94", "np": np, "pd": pd}
exec(compile(src92[src92.index("ALPHAS = "):src92.index("rows = []")], "c92_defs", "exec"), n2)
n2["rng"] = np.random.default_rng(94)
sc = pd.read_csv(ROOT / "results/C11_UCONN_LOGO_2026-10-23_v1/scored_logo.csv.gz", parse_dates=["t"])
cyc = pd.read_csv(ROOT / "tables/C10_UCONN_2026-10-23_v1/uconn_cycling_table.csv.gz")
cyc["t"] = pd.to_datetime(cyc.date)
u = sc.merge(cyc.drop(columns=[c for c in cyc.columns if c in sc.columns and c not in ("cell", "t")]), on=["cell", "t"], how="left")
u["dVa"], u["dVb"] = u.V60 - u.V0, u.V600 - u.V300
gu, yu = u.group.values, u.Q.values
SQ, SL = n2["Stats"](n2["design"](u, True), yu, gu), n2["Stats"](n2["design"](u, False), yu, gu)
zs = []
for h in np.unique(gu):
    if h == k:
        continue
    tr = [j for j in np.unique(gu) if j not in (k, h)]
    pred = SQ.fit(tr, SQ.pick_alpha(tr))(SQ.X[gu == h])
    sig = SL.boot_std(tr, SL.pick_alpha(tr), SL.X[gu == h])
    zs.append(np.quantile(np.abs(pred - yu[gu == h]) / (sig + 1e-3), 0.9))
tau = max(zs)
z = u[u.cell == CELL].copy()
half = tau * (z.boot_std + 1e-3)
z["lo"], z["hi"] = z.MW_quad - half, z.MW_quad + half
z["day"] = (z.t - d.t.min()).dt.total_seconds() / 86400
z[["day", "Q", "MW_quad", "lo", "hi"]].to_csv(OUT / f"overview_cell{CELL}_intervals.csv", index=False)
print("cell", CELL, "group", k, "tests", sum(1 for e in ev if e[1] == 0), "params", best,
      "MAE window", round(float((x.pred - x.Q)[x.kf.notna()].abs().mean()), 4), "MAE kf", round(float((x.kf - x.Q).abs().mean()), 4),
      "tau_D_nested", round(float(tau), 3), "coverage", round(float(((z.Q >= z.lo) & (z.Q <= z.hi)).mean()), 3))
