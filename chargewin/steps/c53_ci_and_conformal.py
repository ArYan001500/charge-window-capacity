"""Cluster-bootstrap confidence intervals and pooled interval baselines."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=False)
rng = np.random.default_rng(53)
sets = {
    "UConn": (R / "C11_UCONN_LOGO_2026-10-23_v1/scored_logo.csv.gz", "MW_quad", "boot_std"),
    "HUST": (R / "C14_HUST_LOGO_2026-10-23_v1/scored_random.csv.gz", "MW_quad", "boot_std"),
    "KIT": (R / "C31_KIT_LOGO_2026-10-24_usage/scored.csv.gz", "MW_quadT", "boot_std"),
}


def eq_mae(df, col):
    return df.groupby("group").apply(lambda z: (z[col] - z.Q).abs().groupby(z.cell).mean().mean()).mean()


def cell_table(df, col):
    return df.assign(ae=(df[col] - df.Q).abs()).groupby(["group", "cell"]).ae.mean().reset_index()


res = {}
for name, (path, mw, gate) in sets.items():
    d = pd.read_csv(path)
    out = {}
    for a, b in ((mw, "M0"), ("P2", "P1")):
        ta, tb = cell_table(d, a).rename(columns={"ae": "a"}), cell_table(d, b).rename(columns={"ae": "b"})
        t = ta.merge(tb, on=["group", "cell"])
        diffs = []
        groups = {g: x for g, x in t.groupby("group")}
        for _ in range(2000):
            m = []
            for g, x in groups.items():
                xs = x.sample(len(x), replace=True, random_state=int(rng.integers(1e9)))
                m.append((xs.a - xs.b).mean())
            diffs.append(np.mean(m))
        out[f"{a}_minus_{b}"] = dict(point=round(float(t.groupby("group").apply(lambda x: (x.a - x.b).mean()).mean()), 4),
                                    CI95=[round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)])
    cov = []
    for g, x in d.groupby("group"):
        src = d[d.group != g]
        q90 = np.quantile((src[mw] - src.Q).abs(), 0.9)
        inside = ((x[mw] - x.Q).abs() <= q90)
        acc = x[gate].rank(pct=True) <= 0.25
        src_acc = src[src.groupby("group")[gate].rank(pct=True) <= 0.25]
        q90a = np.quantile((src_acc[mw] - src_acc.Q).abs(), 0.9)
        cov.append(dict(group=g, coverage=float(inside.mean()), width=float(2 * q90),
                        coverage_acc25=float(((x[acc][mw] - x[acc].Q).abs() <= q90a).mean()), width_acc25=float(2 * q90a)))
    cv = pd.DataFrame(cov)
    out["conformal_90"] = dict(mean_coverage=round(float(cv.coverage.mean()), 3), min_coverage=round(float(cv.coverage.min()), 3),
                               mean_width_Ah=round(float(cv.width.mean()), 4),
                               mean_coverage_acc25=round(float(cv.coverage_acc25.mean()), 3), mean_width_acc25_Ah=round(float(cv.width_acc25.mean()), 4),
                               groups_below_80pct=int((cv.coverage < 0.8).sum()), n_groups=len(cv))
    res[name] = out
json.dump(res, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(res, indent=1))
