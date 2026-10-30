"""UConn: integral and curvature descriptors of slow reference segments."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=False)
DQ = 0.375
c = pd.read_csv(ROOT / "tables/C38_UCONN_REF_CURVES_2026-10-24_v1/uconn_ref_chg_QofV.csv.gz")
QC = [x for x in c.columns if x.startswith("q")]
V = np.array([float(x[1:]) for x in QC])
rpt = pd.read_csv(ROOT / "tables/C10_UCONN_2026-10-23_v1/uconn_rpt_table.csv.gz")[["cell", "wk", "Q_ref"]]
grp = pd.read_csv(ROOT / "data/uconn/cycling_parameters.csv", encoding="utf-8-sig").set_index("Cell ID")["Group Number"]
c = c.merge(rpt, on=["cell", "wk"]).dropna(subset=["Q_ref"]).reset_index(drop=True)
c["group"] = c.cell.map(grp)


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Zs = (X - mu) / sd
    w = np.linalg.solve(Zs.T @ Zs + alpha * np.eye(Zs.shape[1]), Zs.T @ (y - y.mean()))
    return lambda Xn: (Xn - mu) / sd @ w + y.mean()


def pick_alpha(X, y, g):
    return min(np.logspace(-3, 4, 15), key=lambda a: np.mean(
        [np.abs(ridge_fit(X[g != k], y[g != k], a)(X[g == k]) - y[g == k]).mean() for k in np.unique(g)]))


def segment(qv, pos):
    ok = ~np.isnan(qv)
    v, q = V[ok], np.maximum.accumulate(qv[ok])
    if len(v) < 10 or q[-1] - q[0] < DQ + 0.01:
        return None
    if pos == "LOW":
        a = q[0]
    elif pos == "HIGH":
        a = q[-1] - DQ
    else:
        if v[0] > 3.75 or v[-1] < 3.75:
            return None
        a = np.interp(3.75, v, q) - DQ / 2
        if a < q[0] or a + DQ > q[-1]:
            return None
    keep = np.r_[True, np.diff(q) > 1e-9]
    g31 = np.linspace(a, a + DQ, 31)
    return np.interp(g31, q[keep], v[keep])


def feats(vs):
    q = np.linspace(0, DQ, 31)
    v1 = vs[0]
    f0 = list(np.interp(np.linspace(0, DQ, 16), q, vs)) + [v1]
    y = vs - v1
    i1 = np.concatenate([[0], np.cumsum((y[1:] + y[:-1]) / 2 * np.diff(q))])
    i2 = np.concatenate([[0], np.cumsum((i1[1:] + i1[:-1]) / 2 * np.diff(q))])
    i3 = np.concatenate([[0], np.cumsum((i2[1:] + i2[:-1]) / 2 * np.diff(q))])
    sm = np.convolve(vs, np.ones(5) / 5, mode="same")
    sm[:2], sm[-2:] = vs[:2], vs[-2:]
    d2 = np.gradient(np.gradient(sm, q), q)[3:-3]
    f1 = [i1[-1], i2[-1], i3[-1], d2.min(), d2.max(), np.argmin(d2) / len(d2), np.argmax(d2) / len(d2), (vs[-1] - v1) / DQ, v1]
    return f0, f1


res = {}
for pos in ("LOW", "MID", "HIGH"):
    rows = []
    for i, r in c.iterrows():
        s = segment(r[QC].values.astype(float), pos)
        if s is None:
            continue
        f0, f1 = feats(s)
        rows.append(dict(cell=r.cell, group=r.group, Q=r.Q_ref, f0=f0, f1=f1))
    x = pd.DataFrame(rows)
    F0, F1 = np.array(x.f0.tolist()), np.array(x.f1.tolist())
    sets = {"F0": F0, "F1": F1, "F0+F1": np.hstack([F0, F1[:, :-1]])}
    r_pos = dict(n=len(x), cells=int(x.cell.nunique()))
    for name, X in sets.items():
        p = np.full(len(x), np.nan)
        for g in sorted(x.group.unique()):
            tr, te = (x.group != g).values, (x.group == g).values
            p[te] = ridge_fit(X[tr], x.Q.values[tr], pick_alpha(X[tr], x.Q.values[tr], x.group.values[tr]))(X[te])
        x["p"] = p
        r_pos[name] = round(float(x.groupby("group").apply(lambda z: (z.p - z.Q).abs().groupby(z.cell).mean().mean()).mean()), 4)
    res[pos] = r_pos
res["H_pos_pass"] = res["HIGH"]["F0"] < res["MID"]["F0"] and res["HIGH"]["F0"] < res["LOW"]["F0"]
res["H_int_F1_beats_F0"] = f"{sum(res[p]['F1'] < res[p]['F0'] for p in ('LOW', 'MID', 'HIGH'))}/3"
res["H_int_F0F1_beats_F0"] = f"{sum(res[p]['F0+F1'] < res[p]['F0'] for p in ('LOW', 'MID', 'HIGH'))}/3"
res["H_int_pass"] = int(res["H_int_F1_beats_F0"][0]) >= 2 or int(res["H_int_F0F1_beats_F0"][0]) >= 2
json.dump(res, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(res, indent=1))
