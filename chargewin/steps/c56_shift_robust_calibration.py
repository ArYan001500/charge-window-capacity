"""Interval rules A-D calibrated on held-out training groups (not strictly nested)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=False)
sets = {
    "UConn": (R / "C11_UCONN_LOGO_2026-10-23_v1/scored_logo.csv.gz", "MW_quad"),
    "HUST": (R / "C14_HUST_LOGO_2026-10-23_v1/scored_random.csv.gz", "MW_quad"),
    "KIT": (R / "C31_KIT_LOGO_2026-10-24_usage/scored.csv.gz", "MW_quadT"),
}
EPS = 1e-3

res, rows = {}, []
for name, (path, mw) in sets.items():
    d = pd.read_csv(path, usecols=["cell", "group", "Q", mw, "boot_std"])
    d["r"] = (d[mw] - d.Q).abs()
    d["sc"] = d.boot_std + EPS
    d["z"] = d.r / d.sc
    for g, x in d.groupby("group"):
        s = d[d.group != g]
        qa = np.quantile(s.r, 0.9)
        qb = s.groupby("group").r.quantile(0.9).max()
        qc = np.quantile(s.z, 0.9)
        qd = s.groupby("group").z.quantile(0.9).max()
        for m, half in (("A_pooled", np.full(len(x), qa)), ("B_groupmax", np.full(len(x), qb)),
                        ("C_norm", qc * x.sc.values), ("D_norm_groupmax", qd * x.sc.values)):
            rows.append(dict(dataset=name, method=m, group=g, coverage=float((x.r.values <= half).mean()), width=float(2 * half.mean())))
r = pd.DataFrame(rows)
r.to_csv(OUT / "per_group.csv", index=False)
for (ds, m), x in r.groupby(["dataset", "method"]):
    res.setdefault(m, {})[ds] = dict(mean_cov=round(x.coverage.mean(), 3), min_cov=round(x.coverage.min(), 3),
                                     groups_below_80=int((x.coverage < 0.8).sum()), n_groups=len(x), mean_width_Ah=round(x.width.mean(), 4))
for m, v in res.items():
    v["PASS"] = all(z["mean_cov"] >= 0.88 and z["min_cov"] >= 0.80 for z in v.values())
    v["avg_width_over_datasets"] = round(float(np.mean([z["mean_width_Ah"] for k, z in v.items() if isinstance(z, dict)])), 4)
passing = [m for m in res if res[m]["PASS"]]
res["primary"] = min(passing, key=lambda m: res[m]["avg_width_over_datasets"]) if passing else "NONE_PASS"
json.dump(res, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(res, indent=1))
