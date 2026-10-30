"""Strictly nested source prediction banks for the interval rules."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=False)
R = ROOT / "results"
ALPHAS = np.logspace(-3, 4, 15)
EPS = 1e-3
rng = np.random.default_rng(90)
VG = [f"V{s}" for s in range(0, 601, 60)]
W_LIN = VG + ["Q600", "dVa", "dVb", "I0", "I300", "I600"]
W_Q = ["V0", "V600", "Q600", "dVa", "dVb"]


def design(df, quad, temp=False):
    X = df[W_LIN + (["Tmean"] if temp else [])].values.astype(float)
    if quad:
        Z = df[W_Q + (["Tmean"] if temp else [])].values.astype(float)
        iu = np.triu_indices(Z.shape[1])
        X = np.hstack([X, (Z[:, :, None] * Z[:, None, :])[:, iu[0], iu[1]]])
    return X


class Stats:
    """Per-group sufficient statistics of [X, y] for fast standardized ridge fits."""

    def __init__(self, X, y, g):
        self.X, self.y, self.g = X, y, g
        self.groups = np.unique(g)
        self.s = {k: self._sums(X[g == k], y[g == k], None) for k in self.groups}

    @staticmethod
    def _sums(X, y, w):
        w = np.ones(len(X)) if w is None else w
        Xw = X * w[:, None]
        return dict(n=w.sum(), sx=Xw.sum(0), sxx=X.T @ Xw, sy=(w * y).sum(), sxy=Xw.T @ y)

    @staticmethod
    def solve(t, alpha):
        n = t["n"]
        mu = t["sx"] / n
        var = np.diag(t["sxx"]) / n - mu ** 2
        sd = np.sqrt(np.clip(var, 0, None)) + 1e-12
        ybar = t["sy"] / n
        C = t["sxx"] - n * np.outer(mu, mu)
        A = C / np.outer(sd, sd) + alpha * np.eye(len(mu))
        b = (t["sxy"] - n * mu * ybar) / sd
        w = np.linalg.solve(A, b)
        return lambda Xn: (Xn - mu) / sd @ w + ybar

    def total(self, groups):
        ks = list(groups)
        t = {key: sum(self.s[k][key] for k in ks) for key in ("n", "sx", "sxx", "sy", "sxy")}
        return t

    def pick_alpha(self, groups):
        groups = list(groups)
        err = np.zeros(len(ALPHAS))
        for k in groups:
            t = self.total([j for j in groups if j != k])
            Xk, yk = self.X[self.g == k], self.y[self.g == k]
            for i, a in enumerate(ALPHAS):
                err[i] += np.abs(self.solve(t, a)(Xk) - yk).mean()
        return ALPHAS[int(np.argmin(err))]

    def fit(self, groups, alpha):
        return self.solve(self.total(groups), alpha)

    def boot_std(self, groups, alpha, Xt, reps=30):
        m = np.isin(self.g, list(groups))
        Xs, ys = self.X[m], self.y[m]
        preds = []
        for _ in range(reps):
            w = np.bincount(rng.integers(0, len(Xs), len(Xs)), minlength=len(Xs)).astype(float)
            preds.append(self.solve(self._sums(Xs, ys, w), alpha)(Xt))
        return np.std(preds, axis=0)


def load(name):
    if name == "UConn":
        s = pd.read_csv(R / "C11_UCONN_LOGO_2026-10-23_v1/scored_logo.csv.gz", parse_dates=["t"])
        cyc = pd.read_csv(ROOT / "tables/C10_UCONN_2026-10-23_v1/uconn_cycling_table.csv.gz",
                          usecols=["cell", "date", "Q600", "I0", "I300", "I600"] + VG)
        cyc["t"] = pd.to_datetime(cyc.date, format="%Y.%m.%d %H.%M.%S", errors="coerce")
        if cyc.t.isna().mean() > 0.5:
            cyc["t"] = pd.to_datetime(cyc.date, errors="coerce")
        d = s.merge(cyc.dropna(subset=["t"]).drop_duplicates(["cell", "t"]).drop(columns="date"), on=["cell", "t"], how="left")
        return d, "MW_quad", False
    if name == "HUST":
        return pd.read_csv(R / "C14_HUST_LOGO_2026-10-23_v1/scored_random.csv.gz"), "MW_quad", False
    s = pd.read_csv(R / "C31_KIT_LOGO_2026-10-24_usage/scored.csv.gz")
    tc = [f"T{t}" for t in range(0, 601, 30)]
    w = pd.read_csv(ROOT / "tables/C31_KIT_2026-10-24_v1/kit_windows.csv.gz",
                    usecols=["cell", "t", "Q600", "I0", "I300", "I600"] + VG + tc)
    d = s.merge(w, on=["cell", "t"], how="left")
    d["Tmean"] = d[tc].mean(1)
    return d, "MW_quadT", True


res, rows = {}, []
for name in ("UConn", "HUST", "KIT"):
    d, mw, temp = load(name)
    assert d.V600.notna().all() and len(d) > 0, name
    d["dVa"], d["dVb"] = d.V60 - d.V0, d.V600 - d.V300
    y, g = d.Q.values, d.group.values
    SQ = Stats(design(d, True, temp), y, g)
    SL = Stats(design(d, False, False), y, g)
    groups = SQ.groups
    dev = 0.0
    for k in groups:
        src = [j for j in groups if j != k]
        p = SQ.fit(src, SQ.pick_alpha(src))(SQ.X[g == k])
        dev = max(dev, float(np.abs(p - d.loc[g == k, mw].values).max()))
    r_lock = np.abs(d[mw].values - y)
    sc_lock = d.boot_std.values + EPS
    out = {"max_abs_dev_vs_locked_pred": dev}
    for k in groups:
        z_nested, z_old = [], []
        for h in groups:
            if h == k:
                continue
            tr = [j for j in groups if j not in (k, h)]
            Xh = SQ.X[g == h]
            pred = SQ.fit(tr, SQ.pick_alpha(tr))(Xh)
            sig = SL.boot_std(tr, SL.pick_alpha(tr), SL.X[g == h])
            z_nested.append(np.quantile(np.abs(pred - y[g == h]) / (sig + EPS), 0.9))
            z_old.append(np.quantile(r_lock[g == h] / sc_lock[g == h], 0.9))
        for rule, tau in (("D_nested", max(z_nested)), ("D_C56", max(z_old))):
            half = tau * sc_lock[g == k]
            rows.append(dict(dataset=name, rule=rule, group=int(k), coverage=float((r_lock[g == k] <= half).mean()),
                             width=float(2 * half.mean())))
        print(name, k, "done", flush=True)
    res[name] = out
r = pd.DataFrame(rows)
r.to_csv(OUT / "per_group.csv", index=False)
for (ds, rule), x in r.groupby(["dataset", "rule"]):
    res[ds][rule] = dict(mean_cov=round(x.coverage.mean(), 3), min_cov=round(x.coverage.min(), 3),
                         groups_below_80=int((x.coverage < 0.8).sum()), mean_width_Ah=round(x.width.mean(), 4))
res["PASS_nested"] = all(res[ds]["D_nested"]["mean_cov"] >= 0.88 and res[ds]["D_nested"]["min_cov"] >= 0.80
                         for ds in ("UConn", "HUST", "KIT"))
json.dump(res, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(res, indent=1))
