"""UConn: pulse-based estimator with and without window features."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, FEAT, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=False)
TRAIN, TEST = set(range(1, 7)), set(range(7, 12))


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Zs = (X - mu) / sd
    w = np.linalg.solve(Zs.T @ Zs + alpha * np.eye(Zs.shape[1]), Zs.T @ (y - y.mean()))
    return lambda Xn: (Xn - mu) / sd @ w + y.mean()


def pick_alpha(X, y, g):
    return min(np.logspace(-3, 4, 15), key=lambda a: np.mean(
        [np.abs(ridge_fit(X[g != k], y[g != k], a)(X[g == k]) - y[g == k]).mean() for k in np.unique(g)]))


def fitpred(tr, te, cols):
    Xs, Xt = tr[cols].values.astype(float), te[cols].values.astype(float)
    return ridge_fit(Xs, tr.Q.values, pick_alpha(Xs, tr.Q.values, tr.group.values))(Xt)


def metrics(df, col):
    ape = (df[col] - df.Q).abs() / df.Q * 100
    return dict(MAPE_pooled=round(float(ape.mean()), 3), MAPE_eqcell=round(float(ape.groupby(df.cell).mean().mean()), 3),
                R2=round(float(1 - ((df[col] - df.Q) ** 2).sum() / ((df.Q - df.Q.mean()) ** 2).sum()), 3), n=len(df), cells=int(df.cell.nunique()))


F = pd.read_csv(FEAT / "navidi_pulse_features.csv.gz")
base = ["vmin", "vmax", "vmean", "vvar", "vskew", "vkurt", "R0", "R1", "C1", "R2", "C2", "L1"]
F["logC1"], F["logC2"] = np.log(F.C1), np.log(F.C2)
base = [b for b in base if b not in ("C1", "C2")] + ["logC1", "logC2"]
wide = F.pivot_table(index=["cell", "wk", "soc"], columns="typ", values=base).reset_index()
wide.columns = ["cell", "wk", "soc"] + [f"{a}_{b}" for a, b in wide.columns[3:]]
PF = [c for c in wide.columns if c.endswith(("_chg", "_dchg"))]
rpt = pd.read_csv(ROOT / "tables/C10_UCONN_2026-10-23_v1/uconn_rpt_table.csv.gz")
grp = pd.read_csv(ROOT / "data/uconn/cycling_parameters.csv", encoding="utf-8-sig").set_index("Cell ID")["Group Number"]
wide = wide.merge(rpt[["cell", "wk", "Q_ref", "date_start"]], on=["cell", "wk"]).rename(columns={"Q_ref": "Q"}).dropna(subset=PF + ["Q"])
wide["group"] = wide.cell.map(grp)
res = {}
tr, te = wide[wide.group.isin(TRAIN)], wide[wide.group.isin(TEST)].copy()
te["navidi_all"] = fitpred(tr, te, PF)
res["A_their_features_all_pulses"] = metrics(te, "navidi_all")
res["A_reported_GlobalENet"], res["A_reported_best"] = 3.7, 3.1
t = pd.read_csv(ROOT / "results/C11_UCONN_LOGO_2026-10-23_v1/scored_logo.csv.gz", parse_dates=["t"])[["cell", "group", "t", "Q"]]
c = pd.read_csv(ROOT / "tables/C10_UCONN_2026-10-23_v1/uconn_cycling_table.csv.gz")
c["t"] = pd.to_datetime(c.date)
d = t.merge(c, on=["cell", "t"], how="left")
d["dVa"], d["dVb"] = d.V60 - d.V0, d.V600 - d.V300
W = [f"V{s}" for s in range(0, 601, 60)] + ["Q600", "dVa", "dVb", "I0", "I300", "I600"]
rt = rpt[["cell", "wk", "date_start"]].rename(columns={"wk": "rpt_wk"})
rt["t_next"] = pd.to_datetime(rt.date_start)
d = pd.merge_asof(d.sort_values("t"), rt[["cell", "rpt_wk", "t_next"]].sort_values("t_next"), left_on="t", right_on="t_next", by="cell", direction="forward")
one = wide.sample(frac=1.0, random_state=49).groupby(["cell", "wk"]).head(1)[["cell", "wk"] + PF].rename(columns={"wk": "rpt_wk"})
d = d.merge(one, on=["cell", "rpt_wk"], how="inner").dropna(subset=W + PF)
tr, te = d[d.group.isin(TRAIN)], d[d.group.isin(TEST)].copy()
te["pulse_only"] = fitpred(tr, te, PF)
te["pulse_plus_window"] = fitpred(tr, te, PF + W)
res["B_pulse_only_same_rows"] = metrics(te, "pulse_only")
res["B_pulse_plus_window_same_rows"] = metrics(te, "pulse_plus_window")
pc = te.groupby("cell").apply(lambda z: pd.Series({"p": ((z.pulse_only - z.Q).abs() / z.Q).mean(), "pw": ((z.pulse_plus_window - z.Q).abs() / z.Q).mean()}))
res["B_cells_window_helps"] = f"{int((pc.pw < pc.p).sum())}/{len(pc)}"
json.dump(res, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(res, indent=1))
