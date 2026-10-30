"""HUST: fixed-start and masked-start 600-s windows, seven random cell batches, same models as UConn."""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from safe_pickle import safe_loads

ZIP, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=False)
rng = np.random.default_rng(20261023)
GRID = np.arange(0, 601, 10)
z = zipfile.ZipFile(ZIP)
rows = []
for name in sorted(n for n in z.namelist() if n.endswith(".pkl")):
    cell = Path(name).stem
    o = safe_loads(z.read(name))[cell]
    dq = o["dq"]
    cycles = sorted(o["data"].keys())
    for k in cycles[::10]:
        c = o["data"][k]
        ch = c[c.Status.str.contains("charge") & ~c.Status.str.contains("discharge")]
        if len(ch) < 20:
            continue
        t = ch["Time (s)"].values.astype(float)
        t = t - t[0]
        if np.any(np.diff(t) < 0) or t[-1] < 700:
            continue
        v, i = ch["Voltage (V)"].values, ch["Current (mA)"].values / 1000
        q = np.concatenate([[0], np.cumsum(np.diff(t) * 0.5 * (i[1:] + i[:-1]) / 3600)])
        for mode, t0 in (("fixed", 0.0), ("random", float(rng.uniform(0, t[-1] - 600)))):
            g = t0 + GRID
            row = dict(cell=cell, cyc=int(k), mode=mode, Q=float(dq[k]) / 1000, dur=float(t[-1]), t0=t0)
            row.update({f"V{s}": x for s, x in zip(GRID, np.interp(g, t, v))})
            row.update({f"I{s}": x for s, x in zip(GRID, np.interp(g, t, i))})
            row["Q600"] = float(np.interp(g[-1], t, q) - np.interp(g[0], t, q))
            prev = k - 50
            row["Q_prev"] = float(dq[prev]) / 1000 if prev in dq else np.nan
            rows.append(row)
    print(cell, len(rows), flush=True)
d = pd.DataFrame(rows)
d.to_csv(OUT / "hust_windows.csv.gz", index=False)

cells = sorted(d.cell.unique())
perm = rng.permutation(len(cells))
fold_of = {cells[j]: int(f) for f, idx in enumerate(np.array_split(perm, 7)) for j in idx}
d["group"] = d.cell.map(fold_of)
VG = [f"V{s}" for s in range(0, 601, 60)]
d["dVa"], d["dVb"] = d.V60 - d.V0, d.V600 - d.V300
W_LIN = VG + ["Q600", "dVa", "dVb", "I0", "I300", "I600"]
W_Q = ["V0", "V600", "Q600", "dVa", "dVb"]


def design(df, quad):
    X = df[W_LIN].values.astype(float)
    if quad:
        Zq = df[W_Q].values.astype(float)
        iu = np.triu_indices(Zq.shape[1])
        X = np.hstack([X, (Zq[:, :, None] * Zq[:, None, :])[:, iu[0], iu[1]]])
    return X


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Zs = (X - mu) / sd
    w = np.linalg.solve(Zs.T @ Zs + alpha * np.eye(Zs.shape[1]), Zs.T @ (y - y.mean()))
    return lambda Xn: (Xn - mu) / sd @ w + y.mean()


def pick_alpha(X, y, g):
    return min(np.logspace(-3, 4, 15), key=lambda a: np.mean(
        [np.abs(ridge_fit(X[g != k], y[g != k], a)(X[g == k]) - y[g == k]).mean() for k in np.unique(g)]))


res_all = {}
for mode in ("fixed", "random"):
    dm = d[(d["mode"] == mode)].dropna(subset=["Q_prev"]).copy()
    out, shift = [], []
    for held in range(7):
        src, te = dm[dm.group != held], dm[dm.group == held].copy()
        gs = src.group.values
        for quad in (False, True):
            Xs, Xt = design(src, quad), design(te, quad)
            a = pick_alpha(Xs, src.Q.values, gs)
            te["MW_quad" if quad else "MW_lin"] = ridge_fit(Xs, src.Q.values, a)(Xt)
            if not quad:
                te["boot_std"] = np.array([ridge_fit(Xs[ii], src.Q.values[ii], a)(Xt)
                                           for ii in [rng.integers(0, len(src), len(src)) for _ in range(30)]]).std(0)
                drift = (src.Q - src.Q_prev).mean()
                te["P0"], te["P1"] = te.Q_prev, te.Q_prev + drift
                res_src = (src.Q - src.Q_prev - drift).values
                te["P2"] = te.P1 + ridge_fit(Xs, res_src, pick_alpha(Xs, res_src, gs))(Xt)
                mu, sd = Xs.mean(0), Xs.std(0) + 1e-12
                shift.append(dict(group=held, mean_boot_std=float(te.boot_std.mean()),
                                  mean_feature_distance=float(np.linalg.norm(((Xt - mu) / sd).mean(0)))))
        te["M0"] = src.Q.mean()
        out.append(te)
    t = pd.concat(out)
    t.to_csv(OUT / f"scored_{mode}.csv.gz", index=False)
    pols = ["M0", "MW_lin", "MW_quad", "P0", "P1", "P2"]
    pg = pd.DataFrame({p: t.groupby("group").apply(lambda x: (x[p] - x.Q).abs().groupby(x.cell).mean().mean()) for p in pols})
    cov = pd.DataFrame({g: {c: float((x.MW_lin - x.Q).abs()[x.boot_std.rank(method="first") <= c * len(x)].mean())
                            for c in (0.25, 1.0)} for g, x in t.groupby("group")}).T
    sh = pd.DataFrame(shift).set_index("group").join(pg)
    res_all[mode] = dict(
        n_rows=len(t), n_cells=int(t.cell.nunique()),
        primary_mean_over_groups=pg.mean().round(5).to_dict(),
        K1_MW_lin=int((pg.MW_lin < pg.M0).sum()), K1_MW_quad=int((pg.MW_quad < pg.M0).sum()),
        K2_P2_beats_P1=int((pg.P2 < pg.P1).sum()), K3_cov25=int((cov[0.25] < cov[1.0]).sum()),
        K4_spearman={s: float(sh[s].corr(sh.MW_lin, method="spearman")) for s in ["mean_boot_std", "mean_feature_distance"]},
        n_groups=7)
json.dump(res_all, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(res_all, indent=1))
