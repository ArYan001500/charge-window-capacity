"""UConn: capacity read from a 30-min slow charge with a new-cell reference curve."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=False)
rng = np.random.default_rng(60)
c = pd.read_csv(ROOT / "tables/C38_UCONN_REF_CURVES_2026-10-24_v1/uconn_ref_chg_QofV.csv.gz")
QC = [x for x in c.columns if x.startswith("q")]
V = np.array([float(x[1:]) for x in QC])
rpt = pd.read_csv(ROOT / "tables/C10_UCONN_2026-10-23_v1/uconn_rpt_table.csv.gz")[["cell", "wk", "Q_ref"]]
grp = pd.read_csv(ROOT / "data/uconn/cycling_parameters.csv", encoding="utf-8-sig").set_index("Cell ID")["Group Number"]
c = c.merge(rpt, on=["cell", "wk"]).dropna(subset=["Q_ref"])
c["group"] = c.cell.map(grp)
Qmat = c[QC].values
fresh = c.wk == c.groupby("cell").wk.transform("min")


def sheet(g):
    x = c[fresh & (c.group != g)]
    F = x[QC].values / x.Q_ref.values[:, None]
    F = F - np.nanmin(F, axis=1, keepdims=True)
    return np.nanmean(F, axis=0)


def estimate(qv, D, dq):
    ok = ~np.isnan(qv) & ~np.isnan(D)
    v, q, d = V[ok], qv[ok], D[ok]
    q = np.maximum.accumulate(q)
    if q[-1] - q[0] < dq + 1e-3:
        return np.nan, np.nan
    q1 = rng.uniform(q[0], q[-1] - dq)
    v1, v2 = np.interp(q1, q, v), np.interp(q1 + dq, q, v)
    den = np.interp(v2, v, d) - np.interp(v1, v, d)
    return (dq / den if den > 1e-3 else np.nan), v1


res, rows = {}, []
for T in (30, 60):
    dq = 0.75 * T * 60 / 3600
    for g in sorted(c.group.unique()):
        D = sheet(g)
        x = c[c.group == g]
        for i, r in x.iterrows():
            qh, v1 = estimate(Qmat[c.index.get_loc(i)], D, dq)
            rows.append(dict(T=T, group=g, cell=r.cell, wk=r.wk, Q=r.Q_ref, est=qh, v1=v1))
R0 = pd.DataFrame(rows)
R = R0.dropna(subset=["est"]).copy()
R["ae"] = (R.est - R.Q).abs()
R.to_csv(OUT / "estimates.csv", index=False)
for T, x in R.groupby("T"):
    pg = x.groupby("group").apply(lambda z: z.ae.groupby(z.cell).mean().mean())
    young = x[x.Q >= x.Q.median()].ae.mean()
    old = x[x.Q < x.Q.median()].ae.mean()
    res[f"T{T}"] = dict(MAE=round(float(pg.mean()), 4), per_group={int(k): round(float(v), 4) for k, v in pg.items()},
                        bias=round(float((x.est - x.Q).mean()), 4), MAE_young_half=round(float(young), 4), MAE_old_half=round(float(old), 4),
                        n=len(x), n_attempted=int((R0["T"] == T).sum()))
p30 = res["T30"]
res["K_a_pass"] = p30["MAE"] < 0.082 and p30["MAE"] < 0.115
res["K_b_pass"] = p30["MAE"] <= 0.035
c11 = pd.read_csv(ROOT / "results/C11_UCONN_LOGO_2026-10-23_v1/scored_logo.csv.gz", usecols=["cell", "group", "Q", "MW_quad"])
pas = c11.groupby("group").apply(lambda z: (z.MW_quad - z.Q).abs().groupby(z.cell).mean().mean())
res["passive_per_group"] = {int(k): round(float(v), 4) for k, v in pas.items()}
res["K_c_pass"] = all(p30["per_group"][g] < pas[g] for g in (8, 9, 11))
json.dump(res, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(res, indent=1))
