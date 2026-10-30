"""UConn: fixed-voltage pulse blocks of each check-up and the pulse-based reference estimate."""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RPTZIP, C11, C10, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
OUT.mkdir(parents=True, exist_ok=False)
z = zipfile.ZipFile(RPTZIP)
prow = []
for name in sorted(n for n in z.namelist() if n.endswith(".csv")):
    cell = int(name.split("cell_")[1][:2])
    d = pd.read_csv(z.open(name), usecols=["Week Number", "Date (yyyy.mm.dd hh.mm.ss)", "Time (s)", "Voltage (V)",
                                          "Current (A)", "Segment Key", "Pulse Type", "Pulse SOC"])
    d = d.reset_index(drop=True)
    fp = d[d["Segment Key"] == "fastpulse"]
    for (wk, typ, soc), s in fp.groupby(["Week Number", "Pulse Type", "Pulse SOC"]):
        i0 = s.index[0]
        if i0 == 0:
            continue
        vb = d.at[i0 - 1, "Voltage (V)"]
        t = s["Time (s)"].values - s["Time (s)"].values[0]
        v, cur = s["Voltage (V)"].values, s["Current (A)"].values.mean()
        if abs(cur) < 0.5 or len(t) < 10:
            continue
        prow.append(dict(cell=cell, wk=float(wk), typ=typ, soc=int(soc),
                         R1=float((np.interp(1.0, t, v) - vb) / cur), R60=float((v[-1] - vb) / cur),
                         date=s["Date (yyyy.mm.dd hh.mm.ss)"].iloc[0]))
    print(name.rsplit("/", 1)[-1], len(prow), flush=True)
P = pd.DataFrame(prow)
P.to_csv(OUT / "pulse_resistance.csv.gz", index=False)
W = P.pivot_table(index=["cell", "wk"], columns=["typ", "soc"], values=["R1", "R60"])
W.columns = [f"{a}_{b}_{c}" for a, b, c in W.columns]
W = W.reset_index()

rpt = pd.read_csv(C10 / "uconn_rpt_table.csv.gz")
rpt["t_next"] = pd.to_datetime(rpt.date_start)
t11 = pd.read_csv(C11 / "scored_logo.csv.gz", parse_dates=["t"])
cyc = pd.read_csv(C10 / "uconn_cycling_table.csv.gz")
cyc["t"] = pd.to_datetime(cyc.date)
d = t11.merge(cyc, on=["cell", "t"], how="left")
nxt = pd.merge_asof(d.sort_values("t"), rpt[["cell", "t_next", "wk"]].rename(columns={"wk": "rpt_wk"}).sort_values("t_next"),
                    left_on="t", right_on="t_next", by="cell", direction="forward")
d = nxt.merge(W, left_on=["cell", "rpt_wk"], right_on=["cell", "wk"], how="inner", suffixes=("", "_p"))
VG = [f"V{s}" for s in range(0, 601, 60)]
d["dVa"], d["dVb"] = d.V60 - d.V0, d.V600 - d.V300
FW = VG + ["Q600", "dVa", "dVb", "I0", "I300", "I600"]
FQ = ["V0", "V600", "Q600", "dVa", "dVb"]
R50 = [c for c in W.columns if c.endswith("_50") and c.startswith(("R1_", "R60_"))]
RALL = [c for c in W.columns if c.startswith(("R1_", "R60_"))]
d = d.dropna(subset=FW + RALL).copy()


def design(df, cols, quad):
    X = df[cols].values.astype(float)
    if quad:
        Zq = df[FQ].values.astype(float)
        iu = np.triu_indices(Zq.shape[1])
        X = np.hstack([X, (Zq[:, :, None] * Zq[:, None, :])[:, iu[0], iu[1]]])
    return X


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Zs = (X - mu) / sd
    w = np.linalg.solve(Zs.T @ Zs + alpha * np.eye(Zs.shape[1]), Zs.T @ (y - y.mean()))
    return lambda Xn: (Xn - mu) / sd @ w + y.mean()


def pick_alpha(X, y, g):
    return min(np.logspace(-3, 4, 15), key=lambda a: np.mean(
        [np.abs(ridge_fit(X[g != k], y[g != k], a)(X[g == k]) - y[g == k]).mean() for k in np.unique(g)]))


variants = {"W": (FW, True), "W+R50": (FW + R50, True), "W+Rall": (FW + RALL, True), "Rall": (RALL, False)}
out = []
for g in sorted(d.group.unique()):
    s, x = d[d.group != g], d[d.group == g].copy()
    for nm, (cols, quad) in variants.items():
        Xs, Xt = design(s, cols, quad), design(x, cols, quad)
        x[nm] = ridge_fit(Xs, s.Q.values, pick_alpha(Xs, s.Q.values, s.group.values))(Xt)
    out.append(x)
x = pd.concat(out)
x[["cell", "group", "t", "Q"] + list(variants)].to_csv(OUT / "scored.csv.gz", index=False)
pg = pd.DataFrame({p: x.groupby("group").apply(lambda z: (z[p] - z.Q).abs().groupby(z.cell).mean().mean()) for p in variants})
bias = pd.DataFrame({p: x.groupby("group").apply(lambda z: (z[p] - z.Q).mean()) for p in variants})
res = dict(n=len(x), mean_over_groups=pg.mean().round(5).to_dict(),
           beats_W={p: int((pg[p] < pg.W).sum()) for p in variants if p != "W"},
           DoD08_bias={int(k): bias.loc[k].round(3).to_dict() for k in (4, 7, 10, 11)})
pg.round(4).to_csv(OUT / "per_group.csv")
json.dump(res, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(res, indent=1))
print(pg.round(4))
