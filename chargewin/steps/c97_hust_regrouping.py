"""HUST: 50 random regroupings of the cells."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
C14 = ROOT / "results/C14_HUST_LOGO_2026-10-23_v1"
ALPHAS = np.logspace(-3, 4, 15)
EPS = 1e-3
VG = [f"V{s}" for s in range(0, 601, 60)]
W_LIN = VG + ["Q600", "dVa", "dVb", "I0", "I300", "I600"]
W_Q = ["V0", "V600", "Q600", "dVa", "dVb"]


def design(df, quad):
    X = df[W_LIN].values.astype(float)
    if quad:
        Z = df[W_Q].values.astype(float)
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
        sd = np.sqrt(np.clip(np.diag(t["sxx"]) / n - mu ** 2, 0, None)) + 1e-12
        ybar = t["sy"] / n
        C = t["sxx"] - n * np.outer(mu, mu)
        w = np.linalg.solve(C / np.outer(sd, sd) + alpha * np.eye(len(mu)), (t["sxy"] - n * mu * ybar) / sd)
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

    def boot_std(self, groups, alpha, Xt, rng, reps=30):
        m = np.isin(self.g, list(groups))
        Xs, ys = self.X[m], self.y[m]
        preds = []
        for _ in range(reps):
            w = np.bincount(rng.integers(0, len(Xs), len(Xs)), minlength=len(Xs)).astype(float)
            preds.append(self.solve(self._sums(Xs, ys, w), alpha)(Xt))
        return np.std(preds, axis=0)


def eqcell(cells, groups, ae):
    return pd.Series(ae).groupby([groups, cells]).mean().groupby(level=0).mean()


def load(mode):
    d = pd.read_csv(C14 / f"scored_{mode}.csv.gz")
    d["dVa"], d["dVb"] = d.V60 - d.V0, d.V600 - d.V300
    return d


def partition(cells, seed):
    perm = np.random.default_rng(seed).permutation(len(cells))
    return {cells[j]: int(f) for f, idx in enumerate(np.array_split(perm, 7)) for j in idx}


def run(args):
    seed, mode = args
    d = load(mode)
    cells_sorted = sorted(d.cell.unique())
    g = d.group.values if seed == 0 else d.cell.map(partition(cells_sorted, seed)).values
    y, qp, cells = d.Q.values, d.Q_prev.values, d.cell.values
    SQ, SL = Stats(design(d, True), y, g), Stats(design(d, False), y, g)
    SR = Stats(design(d, False), y - qp, g)
    pr = {k: np.full(len(d), np.nan) for k in ("MW_quad", "MW_lin", "M0", "P1", "P2")}
    for k in SQ.groups:
        tr = [j for j in SQ.groups if j != k]
        m, ms = g == k, g != k
        pr["MW_quad"][m] = SQ.fit(tr, SQ.pick_alpha(tr))(SQ.X[m])
        pr["MW_lin"][m] = SL.fit(tr, SL.pick_alpha(tr))(SL.X[m])
        pr["M0"][m] = y[ms].mean()
        pr["P1"][m] = qp[m] + (y[ms] - qp[ms]).mean()
        pr["P2"][m] = qp[m] + SR.fit(tr, SR.pick_alpha(tr))(SR.X[m])
    mae = {p: eqcell(cells, g, np.abs(v - y)) for p, v in pr.items()}
    out = dict(seed=seed, mode=mode, **{f"{p}_MAE": float(v.mean()) for p, v in mae.items()},
               K1_quad=int((mae["MW_quad"] < mae["M0"]).sum()), K1_lin=int((mae["MW_lin"] < mae["M0"]).sum()),
               K2=int((mae["P2"] < mae["P1"]).sum()))
    if seed == 0:
        out["sanity_max_diff"] = float(max(np.abs(pr[p] - d[p].values).max() for p in pr))
    if mode == "random":
        rng = np.random.default_rng(97000 + seed)
        r_lock = np.abs(pr["MW_quad"] - y)
        cov = {"A": [], "D": []}
        wid = {"A": [], "D": []}
        for k in SQ.groups:
            rn, zn = [], []
            for h in SQ.groups:
                if h == k:
                    continue
                tr = [j for j in SQ.groups if j not in (k, h)]
                pred = SQ.fit(tr, SQ.pick_alpha(tr))(SQ.X[g == h])
                sig = SL.boot_std(tr, SL.pick_alpha(tr), SL.X[g == h], rng)
                rn.append(np.abs(pred - y[g == h]))
                zn.append(rn[-1] / (sig + EPS))
            trk = [j for j in SQ.groups if j != k]
            m = g == k
            sc = SL.boot_std(trk, SL.pick_alpha(trk), SL.X[m], rng) + EPS
            for rule, half in (("A", np.full(m.sum(), np.quantile(np.concatenate(rn), 0.9))),
                               ("D", max(np.quantile(z, 0.9) for z in zn) * sc)):
                cov[rule].append(float((r_lock[m] <= half).mean()))
                wid[rule].append(float(2 * half.mean()))
        for rule in "AD":
            out[f"{rule}_mean_cov"] = float(np.mean(cov[rule]))
            out[f"{rule}_min_cov"] = float(np.min(cov[rule]))
            out[f"{rule}_width_Ah"] = float(np.mean(wid[rule]))
    print(seed, mode, flush=True)
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=False)
    jobs = [(s, m) for m in ("random", "fixed") for s in [0] + list(range(1001, 1051))]
    with Pool(12) as p:
        rows = p.map(run, jobs, chunksize=1)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "per_partition.csv", index=False)
    res = {}
    for mode, x in r.groupby("mode"):
        x0, xn = x[x.seed == 0].iloc[0], x[x.seed != 0]
        cols = [c for c in x.columns if c not in ("seed", "mode", "sanity_max_diff") and x[c].notna().any()]
        res[mode] = dict(
            partition0=x0[cols + ["sanity_max_diff"]].to_dict(),
            n_new=len(xn),
            share_K1_quad_7of7=float((xn.K1_quad == 7).mean()), share_K1_lin_7of7=float((xn.K1_lin == 7).mean()),
            share_K2_7of7=float((xn.K2 == 7).mean()),
            K1_quad_min=int(xn.K1_quad.min()), K2_min=int(xn.K2.min()), K2_counts=xn.K2.value_counts().sort_index().to_dict(),
            pct={c: [round(float(v), 4) for v in np.percentile(xn[c], [5, 50, 95])] for c in cols})
        if mode == "random":
            res[mode]["share_D_min_cov_ge_080"] = float((xn.D_min_cov >= 0.80).mean())
            res[mode]["share_A_min_cov_ge_080"] = float((xn.A_min_cov >= 0.80).mean())
            res[mode]["share_D_min_gt_A_min"] = float((xn.D_min_cov > xn.A_min_cov).mean())
    json.dump(res, open(OUT / "summary.json", "w"), indent=2, default=float)
    print(json.dumps(res, indent=1, default=float))
