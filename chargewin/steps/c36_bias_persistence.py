"""Persistence of per-cell window bias and its relation to the fusion gain."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=False)


def persistence(df, interval_col):
    e = df.pred - df.Q
    icc = float(e.groupby(df.cell).transform("mean").var() / e.var())
    im = df.assign(e=e).groupby(["cell", interval_col]).e.mean().reset_index().sort_values(["cell", interval_col])
    pairs = im.groupby("cell").e.apply(lambda s: list(zip(s.values[:-1], s.values[1:])))
    xy = np.array([p for ps in pairs for p in ps])
    lag1 = float(np.corrcoef(xy[:, 0], xy[:, 1])[0, 1]) if len(xy) > 3 else np.nan
    return icc, lag1


def eqcell(df, col):
    return (df[col] - df.Q).abs().groupby(df.cell).mean().mean()


rows = []
u = pd.read_csv(R / "C36_UCONN_MLP_PASSIVE_INPUT_2026-10-24/scored.csv.gz")
for sensor, col, kf_dir in (("ridge", "W_ridge", "C25_BIAS_KF_2026-10-23_v1"), ("MLP", "MLP_src", "C36_UCONN_BIAS_KF_MLP_2026-10-24_v1")):
    kf = pd.read_csv(R / kf_dir / "per_group.csv")
    kf = kf[kf["T"] == 14].set_index("group").mae
    for g, x in u.groupby("group"):
        x = x.assign(pred=x[col])
        icc, lag1 = persistence(x, "Q")
        rows.append(dict(dataset="UConn", sensor=sensor, group=g, passive=eqcell(x, "pred"), kf=float(kf[g]), icc=icc, lag1=lag1))
for sensor, pdir, kdir in (("ridge", "C32_KIT_KALMAN_2026-10-24_v1", "C32_KIT_KALMAN_2026-10-24_v1"),
                           ("MLP", "C35_KIT_MLP_PASSIVE_2026-10-24_v1", "C35_KIT_KALMAN_MLP_2026-10-24_v1")):
    p = pd.read_csv(R / pdir / "window_predictions.csv.gz")
    p = p[p.near]
    kf = pd.read_csv(R / kdir / "per_group.csv")
    kf = kf[kf.k_every == 1].set_index("group").bias_kf
    for g, x in p.groupby("group"):
        icc, lag1 = persistence(x, "Q")
        rows.append(dict(dataset="KIT", sensor=sensor, group=g, passive=eqcell(x, "pred"), kf=float(kf[g]), icc=icc, lag1=lag1))
r = pd.DataFrame(rows)
r["gain"] = r.passive - r.kf
r.to_csv(OUT / "per_group_gain_persistence.csv", index=False)
summ = r.groupby(["dataset", "sensor"]).agg(passive=("passive", "mean"), kf=("kf", "mean"), gain=("gain", "mean"),
                                            kf_wins=("gain", lambda s: int((s > 0).sum())), n=("gain", "size"),
                                            icc=("icc", "median"), lag1=("lag1", "median")).round(4)
corr = {k: float(r.gain.corr(r[k], method="spearman")) for k in ("icc", "lag1", "passive")}
json.dump(dict(per_setting=summ.reset_index().to_dict(orient="records"), spearman_gain_vs=corr, n_points=len(r)),
          open(OUT / "summary.json", "w"), indent=2)
print(summ.to_string())
print("Spearman(gain, .) across", len(r), "group points:", {k: round(v, 3) for k, v in corr.items()})
