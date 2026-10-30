"""Slow 30-min UConn segments: group wins and cluster-bootstrap CIs of the integral and curvature features (F0+F1 vs F0)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=False)
sys.argv = ["c61", str(ROOT), str(OUT / "_unused")]
src = open(Path(__file__).with_name("c61_position_integral_features.py"), encoding="utf-8").read().split("res = {}")[0]
ns = {}
exec(src.replace("OUT.mkdir(parents=True, exist_ok=False)", ""), ns)
c, QC, segment, feats, ridge_fit, pick_alpha = (ns[k] for k in ("c", "QC", "segment", "feats", "ridge_fit", "pick_alpha"))


def predict(x, X):
    p = np.full(len(x), np.nan)
    for g in sorted(x.group.unique()):
        tr, te = (x.group != g).values, (x.group == g).values
        p[te] = ridge_fit(X[tr], x.Q.values[tr], pick_alpha(X[tr], x.Q.values[tr], x.group.values[tr]))(X[te])
    return p


def cell_err(x, col):
    return (x[col] - x.Q).abs().groupby([x.group, x.cell]).mean()


res = {}
for pos in ("LOW", "MID", "HIGH"):
    rows = []
    for _, r in c.iterrows():
        s = segment(r[QC].values.astype(float), pos)
        if s is None:
            continue
        f0, f1 = feats(s)
        rows.append(dict(cell=r.cell, group=r.group, Q=r.Q_ref, f0=f0, f1=f1))
    x = pd.DataFrame(rows)
    F0, F1 = np.array(x.f0.tolist()), np.array(x.f1.tolist())
    x["F0"] = predict(x, F0)
    x["F0F1"] = predict(x, np.hstack([F0, F1[:, :-1]]))
    t = pd.DataFrame({"a": cell_err(x, "F0F1"), "b": cell_err(x, "F0")}).reset_index()
    gm = t.groupby("group")[["a", "b"]].mean()
    rng = np.random.default_rng(61)
    diffs = [np.mean([(z.a - z.b).values[rng.integers(0, len(z), len(z))].mean() for _, z in t.groupby("group")])
             for _ in range(2000)]
    res[pos] = dict(F0=round(float(gm.b.mean()), 4), F0F1=round(float(gm.a.mean()), 4),
                    group_wins=f"{int((gm.a < gm.b).sum())}/{len(gm)}",
                    F0F1_minus_F0_CI95=[round(float(v), 4) for v in np.percentile(diffs, [2.5, 97.5])])
json.dump(res, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(res, indent=1))
