"""KIT and HUST: nested bias-aware fusion with measured tests."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.argv = [sys.argv[0], str(ROOT / "tables/C31_KIT_2026-10-24_v1"), str(OUT / "_unused")]
sys.path.insert(0, str(Path(__file__).parent))
import c32_kit_kalman as K

ALPHAS = np.logspace(-3, 4, 15)


class Stats:
    def __init__(self, X, y, g, train_mask):
        self.X, self.y, self.g, self.m = X, y, g, train_mask
        self.s = {k: self._sums(X[(g == k) & train_mask], y[(g == k) & train_mask]) for k in np.unique(g)}

    @staticmethod
    def _sums(X, y):
        return dict(n=len(X), sx=X.sum(0), sxx=X.T @ X, sy=y.sum(), sxy=X.T @ y)

    @staticmethod
    def solve(t, alpha):
        n = t["n"]
        mu = t["sx"] / n
        sd = np.sqrt(np.clip(np.diag(t["sxx"]) / n - mu ** 2, 0, None)) + 1e-12
        ybar = t["sy"] / n
        A = (t["sxx"] - n * np.outer(mu, mu)) / np.outer(sd, sd) + alpha * np.eye(len(mu))
        w = np.linalg.solve(A, (t["sxy"] - n * mu * ybar) / sd)
        return lambda Xn: (Xn - mu) / sd @ w + ybar

    def total(self, groups):
        return {key: sum(self.s[k][key] for k in groups) for key in ("n", "sx", "sxx", "sy", "sxy")}

    def predict(self, train, Xt):
        train = list(train)
        err = np.zeros(len(ALPHAS))
        for k in train:
            t = self.total([j for j in train if j != k])
            mk = (self.g == k) & self.m
            for i, a in enumerate(ALPHAS):
                err[i] += np.abs(self.solve(t, a)(self.X[mk]) - self.y[mk]).mean()
        return self.solve(self.total(train), ALPHAS[int(np.argmin(err))])(Xt)


def hust_design(df):
    VG = [f"V{s}" for s in range(0, 601, 60)]
    X = df[VG + ["Q600", "dVa", "dVb", "I0", "I300", "I600"]].values.astype(float)
    Z = df[["V0", "V600", "Q600", "dVa", "dVb"]].values.astype(float)
    iu = np.triu_indices(Z.shape[1])
    return np.hstack([X, (Z[:, :, None] * Z[:, None, :])[:, iu[0], iu[1]]])


def load_kit():
    d, lab = K.load()
    labs = {c: lab[lab.cell == c].reset_index(drop=True) for c in d.cell.unique()}
    return d, labs, K.design(d), d.near.values


def load_hust(mode, k_test):
    d = pd.read_csv(ROOT / f"results/C14_HUST_LOGO_2026-10-23_v1/scored_{mode}.csv.gz")
    d = d.sort_values(["cell", "cyc"]).reset_index(drop=True)
    d["t"] = d.cyc.astype(float) * 86400
    first = d.groupby("cell").cyc.transform("min")
    is_test = (d.cyc - first) % k_test == 0
    labs = d[is_test][["cell", "t", "Q"]].rename(columns={"t": "t_lab"})
    d["near"] = ~is_test
    return d, {c: x.reset_index(drop=True) for c, x in labs.groupby("cell")}, hust_design(d), np.ones(len(d), bool)


def eqcell(e, col):
    return e.groupby("group").apply(lambda z: (z[col] - z.Q).abs().groupby(z.cell).mean().mean())


def job(args):
    g, k_every, d, labs, cbg = args
    src = [c for gg, cs in cbg.items() if gg != g for c in cs]
    s = d[d.cell.isin(src) & d.near]
    slope = [np.polyfit(labs[c].t_lab / 86400, labs[c].Q, 1)[0] for c in src if len(labs[c]) >= 2]
    drift = float(np.mean(slope))
    bias = (s.pred - s.Q).groupby(s.cell).mean()
    p_b0 = float(np.var(bias) + np.mean(bias) ** 2)
    r_win = float(np.var((s.pred - s.Q) - s.cell.map(bias)))
    out = dict(group=int(g), k_every=k_every)
    best_src = None
    for name, use_w in (("ref_drift", False), ("bias_kf", True)):
        best, bp = np.inf, None
        for qn in (1e-5, 1e-4, 1e-3):
            for qb in ((1e-5, 1e-4, 1e-3) if use_w else (1e-5,)):
                e = K.evaluate(src, d, labs, k_every, (drift, qn, qb, 1e-4, r_win, p_b0), use_w)
                m = K.score(e)
                if m < best:
                    best, bp, be = m, (qn, qb), e
        if use_w:
            best_src = be
        e = K.evaluate(cbg[g], d, labs, k_every, (drift, bp[0], bp[1], 1e-4, r_win, p_b0), use_w)
        out[name] = float((e.est - e.Q).abs().groupby(e.cell).mean().mean())
        if use_w:
            out["window"] = float((e.pred - e.Q).abs().groupby(e.cell).mean().mean())
    gain_src = (eqcell(best_src, "pred") - eqcell(best_src, "est")).mean()
    out["gate_on"] = bool(gain_src > 0)
    out["gate_correct"] = bool((out["bias_kf"] < out["window"]) == out["gate_on"])
    return out


def run_dataset(tag, d, labs, X, train_mask, locked, k_list, pool):
    y, g = d.Q.values, d.group.values
    S = Stats(X, y, g, train_mask)
    groups = np.unique(g)
    dev = max(float(np.abs(S.predict([j for j in groups if j != k], X[g == k]) - locked[g == k]).max()) for k in groups)
    print(tag, "sanity max |dev|", dev, flush=True)
    assert dev < 1e-5, dev
    cbg = {k: list(np.unique(d.cell[g == k])) for k in groups}
    jobs = []
    for k in groups:
        p = locked.copy()
        for h in groups:
            if h != k:
                p[g == h] = S.predict([j for j in groups if j not in (k, h)], X[g == h])
        dk = d.assign(pred=p)
        jobs += [(k, ke, dk, labs, cbg) for ke in k_list]
        print(tag, "bank", k, flush=True)
    r = pd.DataFrame(pool.map(job, jobs))
    r["dataset"] = tag
    return r, dev


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=False)
    rows, summ = [], {}
    with Pool(16) as pool:
        d, labs, X, tm = load_kit()
        locked = pd.read_csv(ROOT / "results/C32_KIT_KALMAN_2026-10-24_v1/window_predictions.csv.gz")
        d = d.merge(locked[["cell", "t", "pred"]], on=["cell", "t"], how="left")
        assert d.pred.notna().all()
        r, dev = run_dataset("KIT", d, labs, X, tm, d.pred.values, (1, 2), pool)
        rows.append(r)
        r.to_csv(OUT / "per_group_KIT.csv", index=False)
        summ["KIT_sanity_dev"] = dev
        for mode in ("fixed", "random"):
            for k_test, ke in ((50, 1), (50, 2)):
                d, labs, X, tm = load_hust(mode, k_test)
                r, dev = run_dataset(f"HUST_{mode}", d, labs, X, tm, d.MW_quad.values.copy(), (ke,), pool)
                rows.append(r)
                summ[f"HUST_{mode}_sanity_dev"] = dev
    r = pd.concat(rows)
    r.to_csv(OUT / "per_group.csv", index=False)
    for (ds, ke), x in r.groupby(["dataset", "k_every"]):
        summ[f"{ds}_k{ke}"] = dict(ref_drift=round(x.ref_drift.mean(), 4), bias_kf=round(x.bias_kf.mean(), 4),
                                   window=round(x.window.mean(), 4), kf_beats_ref=f"{int((x.bias_kf < x.ref_drift).sum())}/{len(x)}",
                                   kf_beats_window=f"{int((x.bias_kf < x.window).sum())}/{len(x)}",
                                   gate_correct=f"{int(x.gate_correct.sum())}/{len(x)}", gate_on=f"{int(x.gate_on.sum())}/{len(x)}")
    json.dump(summ, open(OUT / "summary.json", "w"), indent=2)
    print(json.dumps(summ, indent=1))
