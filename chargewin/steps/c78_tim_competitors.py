"""Adapted Guo and Shao indicators on the same windows and segments."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=False)
rng = np.random.default_rng(78)
NBIN = 12


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Zs = (X - mu) / sd
    w = np.linalg.solve(Zs.T @ Zs + alpha * np.eye(Zs.shape[1]), Zs.T @ (y - y.mean()))
    return lambda Xn: (Xn - mu) / sd @ w + y.mean()


def pick_alpha(X, y, g):
    return min(np.logspace(-3, 4, 15), key=lambda a: np.mean(
        [np.abs(ridge_fit(X[g != k], y[g != k], a)(X[g == k]) - y[g == k]).mean() for k in np.unique(g)]))


def logo_ridge(X, y, g):
    p = np.full(len(y), np.nan)
    for k in np.unique(g):
        tr, te = g != k, g == k
        p[te] = ridge_fit(X[tr], y[tr], pick_alpha(X[tr], y[tr], g[tr]))(X[te])
    return p


def interp_bins(meas, pred_by_bin, q_by_bin):
    """Guo Eq. 8 analogue: bins ordered by capacity; find the two bins whose predicted indicator brackets the measured one."""
    order = np.argsort(pred_by_bin)
    pv, qv = pred_by_bin[order], q_by_bin[order]
    return float(np.interp(meas, pv, qv))


def cellmae(df, col):
    return df.assign(ae=(df[col] - df.Q).abs()).groupby(["group", "cell"]).ae.mean().reset_index()


def eqmae(df, col):
    t = cellmae(df, col)
    return float(t.groupby("group").ae.mean().mean())


def ci_diff(df, a, b, seed=78):
    ta, tb = cellmae(df, a).rename(columns={"ae": "a"}), cellmae(df, b).rename(columns={"ae": "b"})
    t = ta.merge(tb, on=["group", "cell"])
    r = np.random.default_rng(seed)
    diffs = [np.mean([(lambda z: (z.a - z.b).mean())(x.sample(len(x), replace=True, random_state=int(r.integers(1e9)))) for _, x in t.groupby("group")])
             for _ in range(2000)]
    return [round(float(v), 4) for v in np.percentile(diffs, [2.5, 97.5])]


res = {}
sys.argv = ["c61", str(ROOT), str(OUT / "_unused")]
src61 = open(Path(__file__).with_name("c61_position_integral_features.py"), encoding="utf-8").read().split("res = {}")[0]
src61 = src61.replace("OUT.mkdir(parents=True, exist_ok=False)", "")
ns = {}
exec(src61, ns)
c, QC, V, segment, feats = ns["c"], ns["QC"], ns["V"], ns["segment"], ns["feats"]
grid = V
qmat = []
for _, r in c.iterrows():
    qv = r[QC].values.astype(float)
    ok = ~np.isnan(qv)
    q = np.full(len(V), np.nan)
    q[ok] = np.maximum.accumulate(qv[ok])
    qmat.append(q)
qmat = np.array(qmat)
res["G1_slow_segments"] = {}
for pos in ("LOW", "MID", "HIGH"):
    rows = []
    for i, r in c.iterrows():
        s = segment(r[QC].values.astype(float), pos)
        if s is None:
            continue
        f0, f1 = feats(s)
        rows.append(dict(idx=i, cell=r.cell, group=r.group, Q=r.Q_ref, v1=s[0], v2=s[-1], f0=f0, f1=f1))
    x = pd.DataFrame(rows)
    X = np.hstack([np.array(x.f0.tolist()), np.array(x.f1.tolist())[:, :-1]])
    x["ours"] = logo_ridge(X, x.Q.values, x.group.values)
    x["guo"] = np.nan
    for g in sorted(x.group.unique()):
        srcidx = c.index[c.group != g]
        qs = c.loc[srcidx, "Q_ref"].values
        edges = np.quantile(qs, np.linspace(0, 1, NBIN + 1))
        qb, curves = [], []
        for b in range(NBIN):
            m = (qs >= edges[b]) & (qs <= edges[b + 1])
            cur = np.nanmean(qmat[srcidx[m]], axis=0)
            qb.append(qs[m].mean())
            curves.append(cur)
        qb, curves = np.array(qb), np.array(curves)
        te = x.group == g
        for j in x.index[te]:
            v1, v2 = x.at[j, "v1"], x.at[j, "v2"]
            dq_meas = 0.375
            dq_b = np.array([np.interp(v2, grid[~np.isnan(cv)], cv[~np.isnan(cv)]) - np.interp(v1, grid[~np.isnan(cv)], cv[~np.isnan(cv)]) for cv in curves])
            x.at[j, "guo"] = interp_bins(dq_meas, dq_b, qb)
    res["G1_slow_segments"][pos] = dict(ours_F0F1=round(eqmae(x, "ours"), 4), guo=round(eqmae(x, "guo"), 4),
                                        ours_minus_guo_CI95=ci_diff(x, "ours", "guo"), n=len(x))
res["H1_pass_LOW"] = res["G1_slow_segments"]["LOW"]["ours_minus_guo_CI95"][1] < 0

def windows(name):
    if name == "UConn":
        s = pd.read_csv(ROOT / "results/C11_UCONN_LOGO_2026-10-23_v1/scored_logo.csv.gz", usecols=["cell", "group", "t", "Q", "MW_quad"]).rename(columns={"MW_quad": "ours"})
        cyc = pd.read_csv(ROOT / "tables/C10_UCONN_2026-10-23_v1/uconn_cycling_table.csv.gz").rename(columns={"date": "t"})
        d = s.merge(cyc, on=["cell", "t"], how="inner")
        grp = pd.read_csv(ROOT / "data/uconn/cycling_parameters.csv", encoding="utf-8-sig").set_index("Group Number")["Chg C-Rate 1"]
        d["rate"] = d.group.map(grp.groupby(level=0).first())
        step = 10
    else:
        s = pd.read_csv(ROOT / "results/C31_KIT_LOGO_2026-10-24_usage/scored.csv.gz", usecols=["cell", "group", "t", "Q", "MW_quadT"]).rename(columns={"MW_quadT": "ours"})
        w = pd.read_csv(ROOT / "tables/C31_KIT_2026-10-24_v1/kit_windows.csv.gz")
        d = s.merge(w, on=["cell", "t"], how="inner")
        d["rate"] = d.chg
        step = 30
    ts = list(range(0, 601, step))
    Vm = d[[f"V{t}" for t in ts]].values
    Im = np.abs(d[[f"I{t}" for t in ts]].values)
    P = Vm * Im
    d["E"] = ((P[:, 1:] + P[:, :-1]) / 2 * step / 3600).sum(1)
    return d.dropna(subset=["V0", "V600", "E", "ours"]).reset_index(drop=True)


for name in ("UConn", "KIT"):
    d = windows(name)
    d["guo"] = np.nan
    for g in sorted(d.group.unique()):
        src, te = d[d.group != g], d.index[d.group == g]
        for rate in d.loc[te, "rate"].unique():
            sr = src[np.isclose(src.rate, rate)]
            if len(sr) < 200:
                sr = src
            edges = np.quantile(sr.Q, np.linspace(0, 1, NBIN + 1))
            fits, qb = [], []
            for b in range(NBIN):
                m = (sr.Q >= edges[b]) & (sr.Q <= edges[b + 1])
                xb = sr[m]
                fits.append(np.polyfit(xb.V0, xb.V600, 2))
                qb.append(xb.Q.mean())
            qb = np.array(qb)
            tt = te[np.isclose(d.loc[te, "rate"], rate)]
            for j in tt:
                pred = np.array([np.polyval(f, d.at[j, "V0"]) for f in fits])
                d.at[j, "guo"] = interp_bins(d.at[j, "V600"], pred, qb)
    d["shao_E"] = logo_ridge(np.c_[d.E, d.E ** 2], d.Q.values, d.group.values)
    res[f"G2_S1_{name}"] = dict(ours=round(eqmae(d, "ours"), 4), guo=round(eqmae(d, "guo"), 4), shao_E=round(eqmae(d, "shao_E"), 4),
                                ours_minus_guo_CI95=ci_diff(d, "ours", "guo"), ours_minus_shao_CI95=ci_diff(d, "ours", "shao_E"), n=len(d))
res["H2_pass"] = all(res[f"G2_S1_{n}"]["ours_minus_guo_CI95"][1] < 0 for n in ("UConn", "KIT"))
res["H3_pass"] = all(res[f"G2_S1_{n}"]["ours_minus_shao_CI95"][1] < 0 for n in ("UConn", "KIT"))

rows = []
for i, r in c.iterrows():
    q = qmat[i]
    ok = ~np.isnan(q)
    qq, vv = q[ok] - q[ok][0], grid[ok]
    Qt = r.Q_ref
    def energy(a, b):
        if b > qq[-1]:
            return np.nan
        g2 = np.linspace(a, b, 200)
        return float(np.trapezoid(np.interp(g2, qq, vv), g2))
    rows.append(dict(cell=r.cell, group=r.group, Q=Qt, E_soc=energy(0.3 * Qt, 0.7 * Qt), E_fixed=energy(0.3 * 2.25, 0.3 * 2.25 + 0.4 * 2.25)))
L = pd.DataFrame(rows).dropna()
L["p_soc"] = logo_ridge(np.c_[L.E_soc], L.Q.values, L.group.values)
L["p_fixed"] = logo_ridge(np.c_[L.E_fixed], L.Q.values, L.group.values)
res["S2_leak_UConn"] = dict(SOC_interval_energy=round(eqmae(L, "p_soc"), 4), fixed_segment_energy=round(eqmae(L, "p_fixed"), 4),
                            pearson_SOC=round(float(np.corrcoef(L.E_soc, L.Q)[0, 1]), 4), pearson_fixed=round(float(np.corrcoef(L.E_fixed, L.Q)[0, 1]), 4), n=len(L))
res["H4_pass"] = res["S2_leak_UConn"]["SOC_interval_energy"] <= 0.5 * res["S2_leak_UConn"]["fixed_segment_energy"]
json.dump(res, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(res, indent=1))
