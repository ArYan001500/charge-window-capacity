"""KIT: leave-one-group-out window sensor (usage groups or temperatures held out)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

W_DIR, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SPLIT = sys.argv[3] if len(sys.argv) > 3 else "usage"
OUT.mkdir(parents=True, exist_ok=False)
rng = np.random.default_rng(20261024)
w = pd.read_csv(W_DIR / "kit_windows.csv.gz").sort_values("t")
lab = pd.read_csv(W_DIR / "kit_checkup_labels.csv").sort_values("t_lab")
nxt = pd.merge_asof(w, lab.rename(columns={"t_lab": "t_next"}), left_on="t", right_on="t_next", by="cell", direction="forward")
prv = pd.merge_asof(w[["cell", "t"]], lab.rename(columns={"t_lab": "t_prev", "Q": "Q_prev"}), left_on="t", right_on="t_prev",
                    by="cell", direction="backward")
d = nxt.assign(t_prev=prv.t_prev.values, Q_prev=prv.Q_prev.values)
d = d[(d.t_next - d.t) <= 7 * 86400].dropna(subset=["Q", "Q_prev"]).copy()
d["days_since"] = (d.t - d.t_prev) / 86400
d["days_gap"] = (d.t_next - d.t_prev) / 86400
d = d.sample(frac=1.0, random_state=1).groupby(["cell", "t_next"]).head(10).copy()
if SPLIT == "usage":
    combo = d.apply(lambda r: "WLTP" if r.age_type == 3 else f"{r.vmax}_{r.vmin}_{r.chg}_{r.dis}", axis=1)
else:
    combo = d.temp.astype(str)
d["group"] = combo.astype("category").cat.codes

VG = [f"V{s}" for s in range(0, 601, 60)]
d["dVa"], d["dVb"] = d.V60 - d.V0, d.V600 - d.V300
d["Tmean"] = d[[f"T{s}" for s in range(0, 601, 30)]].mean(1)
W_LIN = VG + ["Q600", "dVa", "dVb", "I0", "I300", "I600"]
W_Q = ["V0", "V600", "Q600", "dVa", "dVb"]


def design(df, quad, temp=False):
    cols = W_LIN + (["Tmean"] if temp else [])
    X = df[cols].values.astype(float)
    if quad:
        Zq = df[W_Q + (["Tmean"] if temp else [])].values.astype(float)
        iu = np.triu_indices(Zq.shape[1])
        X = np.hstack([X, (Zq[:, :, None] * Zq[:, None, :])[:, iu[0], iu[1]]])
    return X


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Zs = (X - mu) / sd
    wv = np.linalg.solve(Zs.T @ Zs + alpha * np.eye(Zs.shape[1]), Zs.T @ (y - y.mean()))
    return lambda Xn: (Xn - mu) / sd @ wv + y.mean()


def pick_alpha(X, y, g):
    return min(np.logspace(-3, 4, 15), key=lambda a: np.mean(
        [np.abs(ridge_fit(X[g != k], y[g != k], a)(X[g == k]) - y[g == k]).mean() for k in np.unique(g)]))


out, shift = [], []
for held in sorted(d.group.unique()):
    src, te = d[d.group != held], d[d.group == held].copy()
    gs = src.group.values
    for nm, quad, temp in (("MW_lin", False, False), ("MW_quad", True, False), ("MW_quadT", True, True)):
        Xs, Xt = design(src, quad, temp), design(te, quad, temp)
        a = pick_alpha(Xs, src.Q.values, gs)
        te[nm] = ridge_fit(Xs, src.Q.values, a)(Xt)
        if nm == "MW_lin":
            te["boot_std"] = np.array([ridge_fit(Xs[ii], src.Q.values[ii], a)(Xt)
                                       for ii in [rng.integers(0, len(src), len(src)) for _ in range(30)]]).std(0)
            drift = ((src.Q - src.Q_prev) / src.days_gap.clip(lower=0.5)).mean()
            te["P0"], te["P1"] = te.Q_prev, te.Q_prev + drift * te.days_since
            res_src = (src.Q - (src.Q_prev + drift * src.days_since)).values
            te["P2"] = te.P1 + ridge_fit(Xs, res_src, pick_alpha(Xs, res_src, gs))(Xt)
            mu, sd = Xs.mean(0), Xs.std(0) + 1e-12
            shift.append(dict(group=held, mean_boot_std=float(te.boot_std.mean()),
                              mean_feature_distance=float(np.linalg.norm(((Xt - mu) / sd).mean(0)))))
    te["M0"] = src.Q.mean()
    out.append(te)
t = pd.concat(out)
t[["cell", "group", "t", "Q", "Q_prev", "M0", "MW_lin", "MW_quad", "MW_quadT", "P0", "P1", "P2", "boot_std"]].to_csv(
    OUT / "scored.csv.gz", index=False)
pols = ["M0", "MW_lin", "MW_quad", "MW_quadT", "P0", "P1", "P2"]
pg = pd.DataFrame({p: t.groupby("group").apply(lambda x: (x[p] - x.Q).abs().groupby(x.cell).mean().mean()) for p in pols})
cov = pd.DataFrame({g: {c: float((x.MW_quad - x.Q).abs()[x.boot_std.rank(method="first") <= c * len(x)].mean())
                        for c in (0.25, 1.0)} for g, x in t.groupby("group")}).T
sh = pd.DataFrame(shift).set_index("group").join(pg)
G = len(pg)
res = dict(split=SPLIT, n_rows=len(t), n_cells=int(t.cell.nunique()), n_groups=G,
           primary_mean_over_groups=pg.mean().round(5).to_dict(),
           K1_MW_quad_beats_M0=int((pg.MW_quad < pg.M0).sum()), K2_P2_beats_P1=int((pg.P2 < pg.P1).sum()),
           K3_cov25_below_full=int((cov[0.25] < cov[1.0]).sum()),
           K4_spearman={s: float(sh[s].corr(sh.MW_quad, method="spearman")) for s in ["mean_boot_std", "mean_feature_distance"]},
           MW_quadT_beats_MW_quad=int((pg.MW_quadT < pg.MW_quad).sum()))
pg.round(4).to_csv(OUT / "per_group.csv")
json.dump(res, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(res, indent=1))
print(pg.round(4))
