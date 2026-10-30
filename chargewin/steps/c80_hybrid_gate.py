"""Learned gate between the window sensor and the adapted Guo lookup."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
NBIN = 12
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


def load(name):
    if name == "UConn":
        s = pd.read_csv(ROOT / "results/C11_UCONN_LOGO_2026-10-23_v1/scored_logo.csv.gz", usecols=["cell", "group", "t", "Q"])
        d = s.merge(pd.read_csv(ROOT / "tables/C10_UCONN_2026-10-23_v1/uconn_cycling_table.csv.gz").rename(columns={"date": "t"}), on=["cell", "t"])
        rate = pd.read_csv(ROOT / "data/uconn/cycling_parameters.csv", encoding="utf-8-sig").groupby("Group Number")["Chg C-Rate 1"].first()
        d["rate"] = d.group.map(rate)
        temp = False
    else:
        s = pd.read_csv(ROOT / "results/C31_KIT_LOGO_2026-10-24_usage/scored.csv.gz", usecols=["cell", "group", "t", "Q"])
        d = s.merge(pd.read_csv(ROOT / "tables/C31_KIT_2026-10-24_v1/kit_windows.csv.gz"), on=["cell", "t"])
        d["rate"] = d.chg
        d["Tmean"] = d[[f"T{x}" for x in range(0, 601, 30)]].mean(1)
        temp = True
    d["dVa"], d["dVb"] = d.V60 - d.V0, d.V600 - d.V300
    d = d.dropna(subset=W_LIN).reset_index(drop=True)
    d["cc"] = ((d.V0 < 4.15) & (d.V600 < 4.18)).astype(int)
    return d, temp


def design(df, temp):
    lin, qd = W_LIN + (["Tmean"] if temp else []), W_Q + (["Tmean"] if temp else [])
    Z = df[qd].values.astype(float)
    iu = np.triu_indices(Z.shape[1])
    return np.hstack([df[lin].values.astype(float), (Z[:, :, None] * Z[:, None, :])[:, iu[0], iu[1]]])


def comp_A(tr, te, temp):
    X, Xt = design(tr, temp), design(te, temp)
    return ridge_fit(X, tr.Q.values, pick_alpha(X, tr.Q.values, tr.group.values))(Xt)


def comp_B(tr, te):
    out = np.full(len(te), np.nan)
    for rate in te.rate.unique():
        sr = tr[np.isclose(tr.rate, rate)]
        if len(sr) < 200:
            sr = tr
        edges = np.quantile(sr.Q, np.linspace(0, 1, NBIN + 1))
        fits, qb = [], []
        for b in range(NBIN):
            xb = sr[(sr.Q >= edges[b]) & (sr.Q <= edges[b + 1])]
            fits.append(np.polyfit(xb.V0, xb.V600, 2))
            qb.append(xb.Q.mean())
        qb = np.array(qb)
        m = np.isclose(te.rate.values, rate)
        for k in np.where(m)[0]:
            pred = np.array([np.polyval(f, te.V0.values[k]) for f in fits])
            o = np.argsort(pred)
            out[k] = np.interp(te.V600.values[k], pred[o], qb[o])
    return out


def fold(args):
    name, g = args
    d, temp = load(name)
    src, tgt = d[d.group != g].copy(), d[d.group == g].copy()
    src["A"], src["B"] = np.nan, np.nan
    for h in sorted(src.group.unique()):
        tr, te = src[src.group != h], src[src.group == h]
        src.loc[te.index, "A"] = comp_A(tr, te, temp)
        src.loc[te.index, "B"] = comp_B(tr, te)
    tgt["A"], tgt["B"] = comp_A(src, tgt, temp), comp_B(src, tgt)
    qe = np.quantile(src.V0, [0.2, 0.4, 0.6, 0.8])
    for x in (src, tgt):
        x["bin"] = x.cc.astype(str) + "_" + np.digitize(x.V0, qe).astype(str)
    choose = {}
    for b, x in src.groupby("bin"):
        choose[b] = "B" if (x.B - x.Q).abs().mean() < (x.A - x.Q).abs().mean() else "A"
    tgt["H1"] = np.where(tgt.bin.map(choose).fillna("A") == "B", tgt.B, tgt.A)
    tgt["H2"] = np.where(tgt.cc == 1, tgt.B, tgt.A)
    S = lambda x: np.c_[x.A, x.B, x.A * x.cc, x.B * x.cc, x.V0]
    tgt["H3"] = ridge_fit(S(src), src.Q.values, 1.0)(S(tgt))
    tgt["routed_B"] = (tgt.bin.map(choose).fillna("A") == "B").astype(int)
    return tgt[["cell", "group", "Q", "cc", "A", "B", "H1", "H2", "H3", "routed_B"]]


def cellmae(df, col):
    return df.assign(ae=(df[col] - df.Q).abs()).groupby(["group", "cell"]).ae.mean().reset_index()


def eqmae(df, col):
    return float(cellmae(df, col).groupby("group").ae.mean().mean())


def ci_diff(df, a, b, seed=80):
    t = cellmae(df, a).rename(columns={"ae": "a"}).merge(cellmae(df, b).rename(columns={"ae": "b"}), on=["group", "cell"])
    r = np.random.default_rng(seed)
    diffs = [np.mean([(lambda z: (z.a - z.b).mean())(x.sample(len(x), replace=True, random_state=int(r.integers(1e9)))) for _, x in t.groupby("group")])
             for _ in range(2000)]
    return [round(float(v), 4) for v in np.percentile(diffs, [2.5, 97.5])]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=False)
    jobs = [(n, g) for n in ("UConn", "KIT") for g in sorted(load(n)[0].group.unique())]
    with Pool(18) as pool:
        parts = pool.map(fold, jobs)
    res = {}
    for n in ("UConn", "KIT"):
        P = pd.concat([p for (nn, _), p in zip(jobs, parts) if nn == n])
        P.to_csv(OUT / f"predictions_{n}.csv.gz", index=False)
        cc = P[P.cc == 1]
        res[n] = dict(A_ours=round(eqmae(P, "A"), 4), B_guo=round(eqmae(P, "B"), 4), H1=round(eqmae(P, "H1"), 4), H2=round(eqmae(P, "H2"), 4),
                      H3=round(eqmae(P, "H3"), 4), H1_minus_A_CI95=ci_diff(P, "H1", "A"), H2_minus_A_CI95=ci_diff(P, "H2", "A"),
                      H3_minus_A_CI95=ci_diff(P, "H3", "A"), share_routed_B=round(float(P.routed_B.mean()), 3), pure_CC_share=round(float(P.cc.mean()), 3),
                      pure_CC_rows={k: round(float((cc[k] - cc.Q).abs().mean()), 4) for k in ("A", "B", "H1")}, n=len(P))
    res["K1_pass"] = res["UConn"]["H1_minus_A_CI95"][1] < 0
    res["K2_pass"] = res["KIT"]["H1_minus_A_CI95"][1] <= 0.002
    json.dump(res, open(OUT / "summary.json", "w"), indent=2)
    print(json.dumps(res, indent=1))
