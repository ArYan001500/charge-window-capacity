"""Error relative to measured capacity, bias by state of health, uncertainty of rho."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=False)
R = ROOT / "results"
SRC = {"UConn": ("C11_UCONN_LOGO_2026-10-23_v1/scored_logo.csv.gz", "MW_quad", 2.25),
       "HUST": ("C14_HUST_LOGO_2026-10-23_v1/scored_random.csv.gz", "MW_quad", 1.1),
       "KIT": ("C31_KIT_LOGO_2026-10-24_usage/scored.csv.gz", "MW_quadT", 3.0)}
BINS, LAB = [0, 0.70, 0.80, 0.90, np.inf], ["<0.70", "0.70-0.80", "0.80-0.90", ">=0.90"]


def eqcell(df, col):
    return float(df.groupby(["group", "cell"])[col].mean().groupby("group").mean().mean())


res = {}
for name, (f, col, nom) in SRC.items():
    d = pd.read_csv(R / f, usecols=["cell", "group", "Q", col])
    d["e"] = d[col] - d.Q
    d["ae"], d["ape"] = d.e.abs(), 100 * d.e.abs() / d.Q
    d["soh"] = pd.cut(d.Q / nom, BINS, labels=LAB, right=False)
    out = dict(MAE_Ah=round(eqcell(d, "ae"), 4), MAE_pct_nominal=round(100 * eqcell(d, "ae") / nom, 2),
               MAPE_pct_measured=round(eqcell(d, "ape"), 2), bins={})
    for b, x in d.groupby("soh", observed=True):
        out["bins"][b] = dict(windows=len(x), cells=int(x.cell.nunique()), MAE=round(float(x.ae.mean()), 4),
                              bias=round(float(x.e.mean()), 4), MAPE_measured=round(float(x.ape.mean()), 2))
    res[name] = out

c91 = pd.read_csv(R / "C91_UCONN_CAUSAL_FUSION_2026-10-29_v1/per_group.csv")
u = c91[(c91.variant == "causal_nested") & (c91["T"] == 14)]
c93 = pd.read_csv(R / "C93_KIT_HUST_NESTED_FUSION_2026-10-29_v1/per_group.csv")
k = c93[(c93.dataset == "KIT") & (c93.k_every == 1)]
pts = pd.concat([pd.DataFrame({"passive": u.window, "gain": u.window - u.kf_bias}),
                 pd.DataFrame({"passive": k.window, "gain": k.window - k.bias_kf})]).reset_index(drop=True)
rho = pts.passive.corr(pts.gain, method="spearman")
rng = np.random.default_rng(101)
bs = [pts.iloc[rng.integers(0, len(pts), len(pts))].pipe(lambda z: z.passive.corr(z.gain, method="spearman")) for _ in range(2000)]
res["spearman"] = dict(n_groups=len(pts), rho=round(float(rho), 3), CI95=[round(float(v), 3) for v in np.nanpercentile(bs, [2.5, 97.5])],
                       negative_gain_groups=int((pts.gain < 0).sum()))
json.dump(res, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(res, indent=1))
