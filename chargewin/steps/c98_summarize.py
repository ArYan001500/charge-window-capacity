"""Summary of the causal Shao runs."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, D = Path(sys.argv[1]), Path(sys.argv[2])


def cellmae(df, col):
    return df.assign(ae=(df[col] - df.Q).abs()).groupby(["group", "cell"]).ae.mean().reset_index()


def eqmae(df, col):
    return float(cellmae(df, col).groupby("group").ae.mean().mean())


def ci_diff(df, a, b, seed=79):
    t = cellmae(df, a).rename(columns={"ae": "a"}).merge(cellmae(df, b).rename(columns={"ae": "b"}), on=["group", "cell"])
    r = np.random.default_rng(seed)
    diffs = [np.mean([(lambda z: (z.a - z.b).mean())(x.sample(len(x), replace=True, random_state=int(r.integers(1e9)))) for _, x in t.groupby("group")])
             for _ in range(2000)]
    return [round(float(v), 4) for v in np.percentile(diffs, [2.5, 97.5])]


lock = {"UConn": ("C11_UCONN_LOGO_2026-10-23_v1/scored_logo.csv.gz", "MW_quad"), "KIT": ("C31_KIT_LOGO_2026-10-24_usage/scored.csv.gz", "MW_quadT")}
res = {}
for name in ("UConn", "KIT"):
    ref = pd.read_csv(ROOT / "results" / lock[name][0], usecols=["cell", "t", lock[name][1]])
    out, preds = {}, {}
    for arch in ("causal", "bi"):
        for s in (0, 1, 2):
            p = pd.read_csv(D / f"pred_{name}_{arch}_s{s}.csv.gz")
            if name == "UConn":
                ref["t"] = ref.t.astype(str)
                p["t"] = p.t.astype(str)
            j = p.merge(ref, on=["cell", "t"], how="left")
            assert j[lock[name][1]].notna().all() and np.abs(j.ours - j[lock[name][1]]).max() < 1e-12
            preds[(arch, s)] = p.shao.values
            out[f"{arch}_s{s}"] = dict(shao=round(eqmae(p, "shao"), 4), ours=round(eqmae(p, "ours"), 4),
                                       ours_minus_shao_CI95=ci_diff(p, "ours", "shao"), n=len(p))
    out["seeds_differ"] = bool(all(np.abs(preds[(a, 0)] - preds[(a, 1)]).max() > 1e-6 for a in ("causal", "bi")))
    for arch in ("causal", "bi"):
        out[f"{arch}_mean_MAE"] = round(float(np.mean([out[f"{arch}_s{s}"]["shao"] for s in (0, 1, 2)])), 4)
        out[f"{arch}_all_CI_exclude0_favour_ours"] = all(out[f"{arch}_s{s}"]["ours_minus_shao_CI95"][1] < 0 for s in (0, 1, 2))
    res[name] = out
json.dump(res, open(D / "summary.json", "w"), indent=2)
print(json.dumps(res, indent=1))
