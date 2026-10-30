"""Adapted Guo lookup against the window sensor on purely constant-current windows (V0 < 4.15 V, V600 < 4.18 V)."""
import json
import sys
from pathlib import Path

import numpy as np

src = open(Path(__file__).with_name("c78_tim_competitors.py"), encoding="utf-8").read()
head = src.split("res = {}")[0]
fn = src[src.index("def windows(name):"):src.index('for name in ("UConn", "KIT"):')]
ns = {}
exec(head + fn, ns)
OUT = ns["OUT"]
res = {}
for name in ("UConn", "KIT"):
    d = ns["windows"](name)
    n_all = len(d)
    d = d[(d.V600 < 4.18) & (d.V0 < 4.15)].reset_index(drop=True)
    d["guo"] = np.nan
    for g in sorted(d.group.unique()):
        src_, te = d[d.group != g], d.index[d.group == g]
        for rate in d.loc[te, "rate"].unique():
            sr = src_[np.isclose(src_.rate, rate)]
            if len(sr) < 200:
                sr = src_
            edges = np.quantile(sr.Q, np.linspace(0, 1, 13))
            fits, qb = [], []
            for b in range(12):
                m = (sr.Q >= edges[b]) & (sr.Q <= edges[b + 1])
                xb = sr[m]
                fits.append(np.polyfit(xb.V0, xb.V600, 2))
                qb.append(xb.Q.mean())
            qb = np.array(qb)
            for j in te[np.isclose(d.loc[te, "rate"], rate)]:
                pred = np.array([np.polyval(f, d.at[j, "V0"]) for f in fits])
                d.at[j, "guo"] = ns["interp_bins"](d.at[j, "V600"], pred, qb)
    res[name] = dict(rows=len(d), share=round(len(d) / n_all, 4), ours=round(float(ns["eqmae"](d, "ours")), 4),
                     guo=round(float(ns["eqmae"](d, "guo")), 4), ours_minus_guo_CI95=ns["ci_diff"](d, "ours", "guo"))
json.dump(res, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(res, indent=1))
