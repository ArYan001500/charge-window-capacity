"""UConn: leave-one-usage-group-out window sensor, prior-aided estimate (Eq. 2) and baselines."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

W_DIR, PARAMS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=False)
rng = np.random.default_rng(20261023)

cyc = pd.read_csv(W_DIR / "uconn_cycling_table.csv.gz")
rpt = pd.read_csv(W_DIR / "uconn_rpt_table.csv.gz")
grp = pd.read_csv(PARAMS, encoding="utf-8-sig").set_index("Cell ID")["Group Number"]
cyc["t"] = pd.to_datetime(cyc.date, format="%Y.%m.%d %H.%M.%S", errors="coerce")
if cyc.t.isna().mean() > 0.5:
    cyc["t"] = pd.to_datetime(cyc.date, errors="coerce")
rpt["t"] = pd.to_datetime(rpt.date_start, errors="coerce")
cyc, rpt = cyc.dropna(subset=["t"]).sort_values("t"), rpt.dropna(subset=["t"]).sort_values("t")

nxt = pd.merge_asof(cyc, rpt[["cell", "t", "Q_ref"]].rename(columns={"t": "t_next", "Q_ref": "Q"}),
                    left_on="t", right_on="t_next", by="cell", direction="forward")
prv = pd.merge_asof(cyc[["cell", "t"]], rpt[["cell", "t", "Q_ref"]].rename(columns={"t": "t_prev", "Q_ref": "Q_prev"}),
                    left_on="t", right_on="t_prev", by="cell", direction="backward")
d = nxt.assign(t_prev=prv.t_prev.values, Q_prev=prv.Q_prev.values)
d = d[(d.t_next - d.t).dt.total_seconds() <= 4 * 86400].dropna(subset=["Q", "Q_prev"]).copy()
d["days_since"] = (d.t - d.t_prev).dt.total_seconds() / 86400
d["days_gap"] = (d.t_next - d.t_prev).dt.total_seconds() / 86400
d = d.sample(frac=1.0, random_state=1).groupby(["cell", "t_next"]).head(10).copy()
d["group"] = d.cell.map(grp)

VG = [f"V{s}" for s in range(0, 601, 60)]
d["dVa"] = d.V60 - d.V0
d["dVb"] = d.V600 - d.V300
W_LIN = VG + ["Q600", "dVa", "dVb"] + [f"I{s}" for s in (0, 300, 600)]
W_Q = ["V0", "V600", "Q600", "dVa", "dVb"]


def design(df, quad):
    X = df[W_LIN].values.astype(float)
    if quad:
        Z = df[W_Q].values.astype(float)
        iu = np.triu_indices(Z.shape[1])
        X = np.hstack([X, (Z[:, :, None] * Z[:, None, :])[:, iu[0], iu[1]]])
    return X


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Zs = (X - mu) / sd
    w = np.linalg.solve(Zs.T @ Zs + alpha * np.eye(Zs.shape[1]), Zs.T @ (y - y.mean()))
    return lambda Xn: (Xn - mu) / sd @ w + y.mean()


def pick_alpha(X, y, g):
    gs = np.unique(g)
    return min(np.logspace(-3, 4, 15), key=lambda a: np.mean(
        [np.abs(ridge_fit(X[g != k], y[g != k], a)(X[g == k]) - y[g == k]).mean() for k in gs]))


out, shift = [], []
for held in sorted(d.group.unique()):
    src, te = d[d.group != held], d[d.group == held].copy()
    gsrc = src.group.values
    for quad in (False, True):
        Xs, Xt = design(src, quad), design(te, quad)
        a = pick_alpha(Xs, src.Q.values, gsrc)
        te[f"MW_{'quad' if quad else 'lin'}"] = ridge_fit(Xs, src.Q.values, a)(Xt)
        if not quad:
            boot = np.array([ridge_fit(Xs[ii], src.Q.values[ii], a)(Xt)
                             for ii in [rng.integers(0, len(src), len(src)) for _ in range(30)]])
            te["boot_std"] = boot.std(0)
            drift = ((src.Q - src.Q_prev) / src.days_gap.clip(lower=0.5)).mean()
            te["P0"] = te.Q_prev
            te["P1"] = te.Q_prev + drift * te.days_since
            src_res = src.Q - (src.Q_prev + drift * src.days_since)
            ar = pick_alpha(Xs, src_res.values, gsrc)
            te["P2"] = te.P1 + ridge_fit(Xs, src_res.values, ar)(Xt)
            mu, sd = Xs.mean(0), Xs.std(0) + 1e-12
            shift.append(dict(group=held, mean_boot_std=float(te.boot_std.mean()),
                              mean_feature_distance=float(np.linalg.norm(((Xt - mu) / sd).mean(0)))))
    te["M0"] = src.Q.mean()
    out.append(te)
t = pd.concat(out)
keep = ["cell", "group", "t", "Q", "Q_prev", "days_since", "M0", "MW_lin", "MW_quad", "P0", "P1", "P2", "boot_std"]
t[keep].to_csv(OUT / "scored_logo.csv.gz", index=False)


def eqcell(x, p):
    return (x[p] - x.Q).abs().groupby(x.cell).mean().mean()


pols = ["M0", "MW_lin", "MW_quad", "P0", "P1", "P2"]
pg = pd.DataFrame({p: t.groupby("group").apply(lambda x: eqcell(x, p)) for p in pols})
cov = {}
for g, x in t.groupby("group"):
    ae = (x.MW_lin - x.Q).abs()
    o = x.boot_std.rank(method="first")
    cov[g] = {c: float(ae[o <= c * len(x)].mean()) for c in (0.25, 0.5, 1.0)}
cov = pd.DataFrame(cov).T
sh = pd.DataFrame(shift).set_index("group").join(pg)
res = dict(
    n_rows=len(t), n_cells=int(t.cell.nunique()),
    primary_mean_over_groups=pg.mean().round(5).to_dict(),
    K1_MW_lin_beats_M0=int((pg.MW_lin < pg.M0).sum()), K1_MW_quad_beats_M0=int((pg.MW_quad < pg.M0).sum()),
    K2_P2_beats_P1=int((pg.P2 < pg.P1).sum()),
    K3_cov25_below_full=int((cov[0.25] < cov[1.0]).sum()),
    K4_spearman={s: float(sh[s].corr(sh.MW_lin, method="spearman")) for s in ["mean_boot_std", "mean_feature_distance"]},
)
pg.round(5).to_csv(OUT / "per_group_eqcell_MAE.csv")
cov.round(5).to_csv(OUT / "per_group_risk_coverage_MW_lin.csv")
sh.round(5).to_csv(OUT / "per_group_shift_scores.csv")
json.dump(res, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(res, indent=1))
print(pg.round(4))
print(cov.round(4))
print(sh.round(4))
