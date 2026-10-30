"""Worst-group coverage uncertainty and width-matched pooled interval."""
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
        return {key: sum(self.s[k][key] for k in groups) for key in ("n", "sx", "sxx", "sy", "sxy")}

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

    def cluster_std(self, groups, alpha, Xt, cells, reps=None):
        reps = CREPS if reps is None else reps
        m = np.isin(self.g, list(groups))
        Xs, ys = self.X[m], self.y[m]
        uc, inv = np.unique(cells[m], return_inverse=True)
        preds = []
        for _ in range(reps):
            cnt = np.bincount(rng_c.integers(0, len(uc), len(uc)), minlength=len(uc)).astype(float)
            preds.append(self.solve(self._sums(Xs, ys, cnt[inv]), alpha)(Xt))
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


def thresholds(r_list, z_list):
    return dict(A=np.quantile(np.concatenate(r_list), 0.9), B=max(np.quantile(r, 0.9) for r in r_list),
                C=np.quantile(np.concatenate(z_list), 0.9), D=max(np.quantile(z, 0.9) for z in z_list))


rows, wins = [], []
res = {}
for name in ("UConn", "HUST", "KIT"):
    d, mw, temp = load(name)
    assert d.V600.notna().all() and len(d) > 0, name
    d["dVa"], d["dVb"] = d.V60 - d.V0, d.V600 - d.V300
    y, g = d.Q.values, d.group.values
    SQ = Stats(design(d, True, temp), y, g)
    SL = Stats(design(d, False, False), y, g)
    groups = SQ.groups
    r_lock = np.abs(d[mw].values - y)
    sc_lock = d.boot_std.values + EPS
    cells = d.cell.values
    for k in groups:
        rn, zn = [], []
        for h in groups:
            if h == k:
                continue
            tr = [j for j in groups if j not in (k, h)]
            pred = SQ.fit(tr, SQ.pick_alpha(tr))(SQ.X[g == h])
            sig = SL.boot_std(tr, SL.pick_alpha(tr), SL.X[g == h])
            rn.append(np.abs(pred - y[g == h]))
            zn.append(rn[-1] / (sig + EPS))
        m = g == k
        hA = np.full(m.sum(), np.quantile(np.concatenate(rn), 0.9))
        hD = max(np.quantile(z, 0.9) for z in zn) * sc_lock[m]
        for rule, half in (("A", hA), ("D", hD)):
            cov = r_lock[m] <= half
            rows.append(dict(dataset=name, rule=rule, group=int(k), coverage=float(cov.mean()),
                             cell_coverage=float(pd.Series(cov).groupby(cells[m]).mean().mean()), width=float(2 * half.mean())))
        wins.append(pd.DataFrame(dict(dataset=name, group=int(k), cell=cells[m].astype(str), r=r_lock[m], hA=hA, hD=hD)))
        print(name, k, flush=True)
r = pd.DataFrame(rows)
w = pd.concat(wins, ignore_index=True)
r.to_csv(OUT / "per_group.csv", index=False)
w.to_csv(OUT / "window_rows.csv.gz", index=False)
c92 = pd.read_csv(R / "C92_NESTED_RULES_AD_2026-10-29_v1/per_group.csv")
c92 = c92[c92.contract == "nested"].set_index(["dataset", "rule", "group"])
rr = r.set_index(["dataset", "rule", "group"])
res["sanity_vs_C92_max"] = float(max((rr[c] - c92.loc[rr.index, c]).abs().max() for c in ("coverage", "cell_coverage", "width")))
res["sanity_pass"] = res["sanity_vs_C92_max"] <= 1e-9

rng_b = np.random.default_rng(101)
for name, x in w.groupby("dataset"):
    out = {}
    byg = {k: [(z.r.values <= z.hA.values, z.r.values <= z.hD.values) for _, z in gx.groupby("cell")] for k, gx in x.groupby("group")}
    pt = {rule: min(np.mean(np.concatenate([c[i] for c in cs])) for cs in byg.values()) for i, rule in enumerate("AD")}
    boots = {"A": [], "D": []}
    for _ in range(2000):
        worst = {"A": 1.0, "D": 1.0}
        for cs in byg.values():
            pick = rng_b.integers(0, len(cs), len(cs))
            for i, rule in enumerate("AD"):
                worst[rule] = min(worst[rule], float(np.mean(np.concatenate([cs[j][i] for j in pick]))))
        for rule in "AD":
            boots[rule].append(worst[rule])
    for rule in "AD":
        out[f"worst_{rule}"] = round(pt[rule], 3)
        out[f"worst_{rule}_CI95"] = [round(float(v), 3) for v in np.percentile(boots[rule], [2.5, 97.5])]
        out[f"share_boot_worst_{rule}_ge_080"] = round(float(np.mean(np.array(boots[rule]) >= 0.80)), 3)
    wA = x.groupby("group").hA.mean().mean()
    wD = x.groupby("group").hD.mean().mean()
    c = wD / wA
    gc = x.assign(cov=x.r <= c * x.hA).groupby("group")["cov"].mean()
    out.update(width_factor=round(float(c), 3), scaledA_worst=round(float(gc.min()), 3), scaledA_mean=round(float(gc.mean()), 3),
               D_mean_width=round(float(2 * wD), 4), scaledA_mean_width=round(float(2 * c * wA), 4))
    res[name] = out
json.dump(res, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(res, indent=1))
