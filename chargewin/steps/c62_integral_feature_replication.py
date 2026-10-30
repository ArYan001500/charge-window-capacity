"""Integral and curvature descriptors added to the window sensor on all datasets."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
R = ROOT / "results"
VG = [f"V{s}" for s in range(0, 601, 60)]
W_LIN = VG + ["Q600", "dVa", "dVb", "I0", "I300", "I600"]
W_Q = ["V0", "V600", "Q600", "dVa", "dVb"]


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Zs = (X - mu) / sd
    w = np.linalg.solve(Zs.T @ Zs + alpha * np.eye(Zs.shape[1]), Zs.T @ (y - y.mean()))
    return lambda Xn: (Xn - mu) / sd @ w + y.mean()


def pick_alpha(X, y, g):
    return min(np.logspace(-3, 4, 15), key=lambda a: np.mean(
        [np.abs(ridge_fit(X[g != k], y[g != k], a)(X[g == k]) - y[g == k]).mean() for k in np.unique(g)]))


def design(df, temp):
    lin = W_LIN + (["Tmean"] if temp else [])
    qd = W_Q + (["Tmean"] if temp else [])
    Z = df[qd].values.astype(float)
    iu = np.triu_indices(Z.shape[1])
    return np.hstack([df[lin].values.astype(float), (Z[:, :, None] * Z[:, None, :])[:, iu[0], iu[1]]])


def f1(df, step):
    ts = list(range(0, 601, step))
    Vm = df[[f"V{s}" for s in ts]].values.astype(float)
    Im = df[[f"I{s}" for s in ts]].values.astype(float)
    out = np.full((len(df), 8), np.nan)
    for k in range(len(df)):
        v, i = Vm[k], np.abs(Im[k])
        q = np.concatenate([[0], np.cumsum((i[1:] + i[:-1]) / 2 * step / 3600)])
        keep = np.r_[True, np.diff(q) > 1e-7]
        if keep.sum() < 6 or q[-1] < 1e-3:
            continue
        qq = np.linspace(0, q[-1], 31)
        vs = np.interp(qq, q[keep], v[keep])
        y = vs - vs[0]
        dq = np.diff(qq)
        i1 = np.concatenate([[0], np.cumsum((y[1:] + y[:-1]) / 2 * dq)])
        i2 = np.concatenate([[0], np.cumsum((i1[1:] + i1[:-1]) / 2 * dq)])
        i3 = np.concatenate([[0], np.cumsum((i2[1:] + i2[:-1]) / 2 * dq)])
        sm = np.convolve(vs, np.ones(5) / 5, mode="same")
        sm[:2], sm[-2:] = vs[:2], vs[-2:]
        d2 = np.gradient(np.gradient(sm, qq), qq)[3:-3]
        out[k] = [i1[-1], i2[-1], i3[-1], d2.min(), d2.max(), np.argmin(d2) / len(d2), np.argmax(d2) / len(d2), (vs[-1] - vs[0]) / q[-1]]
    return out


def load(name):
    if name == "UConn":
        s = pd.read_csv(R / "C11_UCONN_LOGO_2026-10-23_v1/scored_logo.csv.gz", usecols=["cell", "group", "t", "Q", "MW_quad"])
        c = pd.read_csv(ROOT / "tables/C10_UCONN_2026-10-23_v1/uconn_cycling_table.csv.gz").rename(columns={"date": "t"})
        d = s.merge(c, on=["cell", "t"], how="inner")
        step, temp, ref = 10, False, "MW_quad"
    elif name.startswith("HUST"):
        d = pd.read_csv(R / f"C14_HUST_LOGO_2026-10-23_v1/scored_{name.split('_')[1]}.csv.gz")
        step, temp, ref = 10, False, "MW_quad"
    else:
        s = pd.read_csv(R / "C31_KIT_LOGO_2026-10-24_usage/scored.csv.gz", usecols=["cell", "group", "t", "Q", "MW_quadT"])
        w = pd.read_csv(ROOT / "tables/C31_KIT_2026-10-24_v1/kit_windows.csv.gz")
        d = s.merge(w, on=["cell", "t"], how="inner")
        d["Tmean"] = d[[f"T{x}" for x in range(0, 601, 30)]].mean(1)
        step, temp, ref = 30, True, "MW_quadT"
    d["dVa"], d["dVb"] = d.V60 - d.V0, d.V600 - d.V300
    F = f1(d, step)
    ok = ~np.isnan(F).any(1) & d[W_LIN].notna().all(1).values
    return d[ok].reset_index(drop=True), F[ok], temp, ref


def run(name):
    d, F, temp, ref = load(name)
    X0 = design(d, temp)
    X1 = np.hstack([X0, F])
    for col, X in (("base", X0), ("base_F1", X1)):
        p = np.full(len(d), np.nan)
        for g in sorted(d.group.unique()):
            tr, te = (d.group != g).values, (d.group == g).values
            p[te] = ridge_fit(X[tr], d.Q.values[tr], pick_alpha(X[tr], d.Q.values[tr], d.group.values[tr]))(X[te])
        d[col] = np.abs(p - d.Q.values)
    d["ref_ae"] = (d[ref] - d.Q).abs()
    t = d.groupby(["group", "cell"])[["base", "base_F1", "ref_ae"]].mean().reset_index()
    pg = t.groupby("group")[["base", "base_F1", "ref_ae"]].mean()
    rng = np.random.default_rng(62)
    diffs = [np.mean([(lambda z: (z.base_F1 - z.base).mean())(x.sample(len(x), replace=True, random_state=int(rng.integers(1e9))))
                      for _, x in t.groupby("group")]) for _ in range(2000)]
    ci = np.percentile(diffs, [2.5, 97.5])
    return name, dict(n=len(d), published_ref=round(float(pg.ref_ae.mean()), 4), base=round(float(pg.base.mean()), 4),
                      base_F1=round(float(pg.base_F1.mean()), 4), wins=f"{int((pg.base_F1 < pg.base).sum())}/{len(pg)}",
                      CI95=[round(float(ci[0]), 4), round(float(ci[1]), 4)], better=bool(ci[1] < 0), worse=bool(ci[0] > 0))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=False)
    with Pool(4) as pool:
        res = dict(pool.map(run, ["UConn", "HUST_fixed", "HUST_random", "KIT"]))
    res["H_rep_pass"] = sum(v["better"] for v in res.values()) >= 2 and not any(v["worse"] for v in res.values() if isinstance(v, dict))
    json.dump(res, open(OUT / "summary.json", "w"), indent=2)
    print(json.dumps(res, indent=1))
