"""UConn filter with an AR(1) window error in the state."""
import pickle
from multiprocessing import Pool
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
R = ROOT / "results"
C10 = ROOT / "tables/C10_UCONN_2026-10-23_v1"
ALPHAS = np.logspace(-3, 4, 15)
TS = (7, 14, 28, 56)
VG = [f"V{s}" for s in range(0, 601, 60)]
FW = VG + ["Q600", "dVa", "dVb", "I0", "I300", "I600"]
FQ = ["V0", "V600", "Q600", "dVa", "dVb"]


class Stats:
    def __init__(self, X, y, g):
        self.X, self.y, self.g = X, y, g
        self.s = {k: self._sums(X[g == k], y[g == k]) for k in np.unique(g)}

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
            Xk, yk = self.X[self.g == k], self.y[self.g == k]
            for i, a in enumerate(ALPHAS):
                err[i] += np.abs(self.solve(t, a)(Xk) - yk).mean()
        return self.solve(self.total(train), ALPHAS[int(np.argmin(err))])(Xt)


def design(df, cols, quad):
    X = df[cols].values.astype(float)
    if quad:
        Z = df[FQ].values.astype(float)
        iu = np.triu_indices(Z.shape[1])
        X = np.hstack([X, (Z[:, :, None] * Z[:, None, :])[:, iu[0], iu[1]]])
    return X


def prepare():
    global d, y, g, groups, nest, dev, day, tday, cell_rows, cells_by_group, slope
    P = pd.read_csv(R / "C19_UCONN_PULSE_2026-10-23_v1/pulse_resistance.csv.gz")
    Wp = P.pivot_table(index=["cell", "wk"], columns=["typ", "soc"], values=["R1", "R60"])
    Wp.columns = [f"{a}_{b}_{c}" for a, b, c in Wp.columns]
    Wp = Wp.reset_index()
    rpt = pd.read_csv(C10 / "uconn_rpt_table.csv.gz")
    rpt["t_next"] = pd.to_datetime(rpt.date_start)
    t11 = pd.read_csv(R / "C11_UCONN_LOGO_2026-10-23_v1/scored_logo.csv.gz", parse_dates=["t"])
    cyc = pd.read_csv(C10 / "uconn_cycling_table.csv.gz")
    cyc["t"] = pd.to_datetime(cyc.date)
    d = t11.merge(cyc, on=["cell", "t"], how="left")
    d = pd.merge_asof(d.sort_values("t"), rpt[["cell", "t_next", "wk"]].rename(columns={"wk": "rpt_wk"}).sort_values("t_next"),
                      left_on="t", right_on="t_next", by="cell", direction="forward")
    d = d.merge(Wp, left_on=["cell", "rpt_wk"], right_on=["cell", "wk"], how="inner", suffixes=("", "_p"))
    d["dVa"], d["dVb"] = d.V60 - d.V0, d.V600 - d.V300
    RALL = [c for c in Wp.columns if c.startswith(("R1_", "R60_"))]
    d = d.dropna(subset=FW + RALL).sort_values(["cell", "t"]).reset_index(drop=True)
    lock = pd.read_csv(R / "C19_UCONN_PULSE_2026-10-23_v1/scored.csv.gz", parse_dates=["t"])
    d = d.merge(lock[["cell", "t", "W", "W+Rall", "Rall"]].rename(columns={"W": "L_W", "W+Rall": "L_WR", "Rall": "L_R"}), on=["cell", "t"])
    assert len(d) == 12900 and (d.t < d.t_next).all(), len(d)
    y, g = d.Q.values, d.group.values
    groups = np.unique(g)
    MODELS = {"W": Stats(design(d, FW, True), y, g), "WR": Stats(design(d, FW + RALL, True), y, g),
              "R": Stats(design(d, RALL, False), y, g)}

    dev = {}
    for m, S in MODELS.items():
        dev[m] = max(float(np.abs(S.predict([j for j in groups if j != k], S.X[g == k]) - d.loc[g == k, "L_" + m]).max())
                     for k in groups)
    print("sanity max |dev| vs locked:", dev, flush=True)
    assert max(dev.values()) < 1e-6

    nest = {m: {} for m in MODELS}
    for k in groups:
        for m, S in MODELS.items():
            p = np.full(len(d), np.nan)
            for h in groups:
                if h != k:
                    p[g == h] = S.predict([j for j in groups if j not in (k, h)], S.X[g == h])
            nest[m][k] = p
        print("nested bank", k, flush=True)

    day = ((d.t - d.t.min()).dt.total_seconds() / 86400).values
    tday = ((d.t_next - d.t.min()).dt.total_seconds() / 86400).values
    cell_rows = {c: np.where(d.cell.values == c)[0] for c in np.unique(d.cell)}
    cells_by_group = {k: list(np.unique(d.cell[g == k])) for k in groups}
    slope = {c: np.polyfit(day[ix], y[ix], 1)[0] for c, ix in cell_rows.items()}


def load_state(path):
    global d, y, g, groups, nest, dev, day, tday, cell_rows, cells_by_group, slope
    with open(path, "rb") as f:
        st = pickle.load(f)
    d, y, g, groups, nest, dev, day, tday, cell_rows, cells_by_group, slope = (st[k] for k in (
        "d", "y", "g", "groups", "nest", "dev", "day", "tday", "cell_rows", "cells_by_group", "slope"))


def events(c, wpred, tpred, T):
    """Causal event stream of one cell: tests at check-up start times (value = test model on the last paired window)."""
    ix = cell_rows[c]
    rp = pd.DataFrame({"tk": tday[ix], "v": tpred[ix], "i": ix}).groupby("tk").last().reset_index()
    chosen, last = [], -np.inf
    for tk, v in zip(rp.tk, rp.v):
        if tk - last >= T:
            chosen.append((tk, 0, v))
            last = tk
    ev = chosen + [(day[i], 1, i) for i in ix]
    ev.sort(key=lambda e: (e[0], e[1]))
    return ev, wpred


def run(c, wpred, tpred, T, mode, drift, qn, qb, rt, rw, pb0):
    ev, _ = events(c, wpred, tpred, T)
    out = {}
    Q = b = p11 = p12 = p22 = None
    prev = None
    for t, kind, v in ev:
        if Q is not None:
            dt = t - prev
            Q += drift * dt
            p11 += qn * dt
            p22 += qb * dt
        prev = t
        if kind == 0:
            if mode == "ref":
                Q, b, p11, p12, p22 = v, 0.0, rt, 0.0, pb0
            elif Q is None:
                Q, b, p11, p12, p22 = v, 0.0, rt, 0.0, pb0
            else:
                S = p11 + rt
                k1, k2 = p11 / S, p12 / S
                inn = v - Q
                Q, b = Q + k1 * inn, b + k2 * inn
                p11, p12, p22 = p11 - k1 * p11, p12 - k1 * p12, p22 - k2 * p12
        else:
            if Q is None:
                continue
            z = wpred[v]
            if mode == "bias":
                h1, h2 = p11 + p12, p12 + p22
                S = h1 + h2 + rw
                k1, k2 = h1 / S, h2 / S
                inn = z - (Q + b)
                Q, b = Q + k1 * inn, b + k2 * inn
                p11, p12, p22 = p11 - k1 * h1, p12 - k1 * h2, p22 - k2 * h2
            elif mode == "scalar":
                S = p11 + rw
                k1 = p11 / S
                Q += k1 * (z - Q)
                p11 -= k1 * p11
            out[v] = Q if mode != "window" else z
    return out


def cell_mae(est):
    ix = np.array(list(est))
    return float(np.mean(np.abs(np.array(list(est.values())) - y[ix])))


def group_mae(cells, ests):
    return float(np.mean([cell_mae(ests[c]) for c in cells]))


def fit_params(src_cells, wbank, tbank):
    rows = np.concatenate([cell_rows[c] for c in src_cells])
    ew = wbank[rows] - y[rows]
    cb = pd.Series(ew).groupby(d.cell.values[rows]).mean()
    return dict(drift=float(np.mean([slope[c] for c in src_cells])), rt=float(np.var(tbank[rows] - y[rows])),
                pb0=float(np.var(cb) + np.mean(cb) ** 2),
                rw_in=float(np.var(ew - d.cell.iloc[rows].map(cb).values)), rw_all=float(np.var(ew)))


def run_ar(c, wpred, tpred, T, drift, qn, qb, rt, rw_in, pb0, tau, rho):
    """[Q, b, e]: e is an AR(1) window error, phi = exp(-dt/tau), stationary variance rho*rw_in; white part (1-rho)*rw_in."""
    ev, _ = events(c, wpred, tpred, T)
    out = {}
    x = P = None
    prev = None
    se2, rw = rho * rw_in, (1 - rho) * rw_in
    Ht, Hw = np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
    for t, kind, v in ev:
        if x is not None:
            dt = t - prev
            phi = np.exp(-dt / tau)
            F = np.diag([1.0, 1.0, phi])
            x = F @ x + np.array([drift * dt, 0.0, 0.0])
            P = F @ P @ F.T + np.diag([qn * dt, qb * dt, se2 * (1 - phi ** 2)])
        prev = t
        if kind == 0:
            if x is None:
                x, P = np.array([v, 0.0, 0.0]), np.diag([rt, pb0, se2])
            else:
                S = Ht @ P @ Ht + rt
                K = P @ Ht / S
                x = x + K * (v - Ht @ x)
                P = P - np.outer(K, Ht @ P)
        else:
            if x is None:
                continue
            S = Hw @ P @ Hw + rw
            K = P @ Hw / S
            x = x + K * (wpred[v] - Hw @ x)
            P = P - np.outer(K, Hw @ P)
            out[v] = x[0]
    return out


QGRID = [(qn, qb) for qn in (1e-5, 1e-4, 1e-3) for qb in (1e-6, 1e-5, 1e-4, 1e-3)]
TAUS, RHOS = (0.5, 2.0, 7.0, 28.0), (0.0, 0.25, 0.5, 0.75)


def job(args):
    T, k = args
    src = [c for kk, cs in cells_by_group.items() if kk != k for c in cs]
    sw, st = nest["W"][k], nest["WR"][k]
    pr = fit_params(src, sw, st)
    src_groups = [kk for kk in groups if kk != k]
    sc_b = []
    for qn, qb in QGRID:
        e = {c: run(c, sw, st, T, "bias", pr["drift"], qn, qb, pr["rt"], pr["rw_in"], pr["pb0"]) for c in src}
        sc_b.append((np.mean([group_mae(cells_by_group[kk], e) for kk in src_groups]), qn, qb))
    bb = min(sc_b, key=lambda s: s[0])
    sc_a = []
    for qn, qb in QGRID:
        for tau in TAUS:
            for rho in RHOS:
                e = {c: run_ar(c, sw, st, T, pr["drift"], qn, qb, pr["rt"], pr["rw_in"], pr["pb0"], tau, rho) for c in src}
                sc_a.append((np.mean([group_mae(cells_by_group[kk], e) for kk in src_groups]), qn, qb, tau, rho))
    ba = min(sc_a, key=lambda s: s[0])
    tw, tt = d.L_W.values, d.L_WR.values
    tc = cells_by_group[k]
    est = {"ref_drift": {c: run(c, tw, tt, T, "ref", pr["drift"], 0, 0, pr["rt"], 1, pr["pb0"]) for c in tc},
           "kf_bias": {c: run(c, tw, tt, T, "bias", pr["drift"], bb[1], bb[2], pr["rt"], pr["rw_in"], pr["pb0"]) for c in tc},
           "bias_ar": {c: run_ar(c, tw, tt, T, pr["drift"], ba[1], ba[2], pr["rt"], pr["rw_in"], pr["pb0"], ba[3], ba[4])
                       for c in tc},
           "ar_rho0": {c: run_ar(c, tw, tt, T, pr["drift"], bb[1], bb[2], pr["rt"], pr["rw_in"], pr["pb0"], 1.0, 0.0)
                       for c in tc}}
    r = dict(T=T, group=int(k), qn_b=bb[1], qb_b=bb[2], qn_a=ba[1], qb_a=ba[2], tau=ba[3], rho=ba[4],
             src_score_bias=bb[0], src_score_ar=ba[0], **{m: group_mae(tc, e) for m, e in est.items()})
    cells = [dict(T=T, group=int(k), cell=int(c), **{m: cell_mae(est[m][c]) for m in est}) for c in tc]
    print("T", T, "group", k, flush=True)
    return r, cells


def boot_ci(cr, a, b, n=2000, seed=91):
    rng = np.random.default_rng(seed)
    diffs = []
    byg = {k: x[a].values - x[b].values for k, x in cr.groupby("group")}
    for _ in range(n):
        diffs.append(np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in byg.values()]))
    return [round(float(np.quantile(diffs, 0.025)), 4), round(float(np.quantile(diffs, 0.975)), 4)]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=False)
    prepare()
    st = dict(d=d, y=y, g=g, groups=groups, nest=nest, dev=dev, day=day, tday=tday, cell_rows=cell_rows,
              cells_by_group=cells_by_group, slope=slope)
    with open(OUT / "state.pkl", "wb") as f:
        pickle.dump(st, f)
    with Pool(16, initializer=load_state, initargs=(OUT / "state.pkl",)) as p:
        res = p.map(job, [(T, k) for T in TS for k in groups], chunksize=1)
    (OUT / "state.pkl").unlink()
    pg = pd.DataFrame([r for r, _ in res])
    pc = pd.DataFrame([c for _, cs in res for c in cs])
    pg.to_csv(OUT / "per_group.csv", index=False)
    pc.to_csv(OUT / "per_cell.csv", index=False)
    c91 = pd.read_csv(R / "C91_UCONN_CAUSAL_FUSION_2026-10-29_v1/per_group.csv")
    c91 = c91[c91.variant == "causal_nested"].set_index(["T", "group"])
    q = pg.set_index(["T", "group"])
    summ = dict(sanity_locked_dev=dev,
                S100_rho0_vs_kf_bias_max=float((q.ar_rho0 - q.kf_bias).abs().max()),
                S100_kf_bias_vs_C91_max=float((q.kf_bias - c91.loc[q.index, "kf_bias"]).abs().max()),
                S100_ref_vs_C91_max=float((q.ref_drift - c91.loc[q.index, "ref_drift"]).abs().max()))
    summ["S100_pass"] = max(v for k, v in summ.items() if k.startswith("S100")) <= 1e-9
    for T in TS:
        x, c = pg[pg["T"] == T], pc[pc["T"] == T]
        summ[f"T{T}"] = dict(**{m: round(float(x[m].mean()), 4) for m in ("ref_drift", "kf_bias", "bias_ar")},
                             ar_beats_ref_groups=int((x.bias_ar < x.ref_drift).sum()),
                             bias_beats_ref_groups=int((x.kf_bias < x.ref_drift).sum()),
                             ar_beats_bias_groups=int((x.bias_ar < x.kf_bias).sum()),
                             ci_ar_minus_bias=boot_ci(c, "bias_ar", "kf_bias"), ci_ar_minus_ref=boot_ci(c, "bias_ar", "ref_drift"),
                             selected_tau={str(a): int(b) for a, b in x.tau.value_counts().items()},
                             selected_rho={str(a): int(b) for a, b in x.rho.value_counts().items()})
    json.dump(summ, open(OUT / "summary.json", "w"), indent=2, default=float)
    print(json.dumps(summ, indent=1, default=float))
