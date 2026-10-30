"""All windows scored by days to the next test (UConn, KIT)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=False)
R = ROOT / "results"
ALPHAS = np.logspace(-3, 4, 15)
VG = [f"V{s}" for s in range(0, 601, 60)]
W_LIN = VG + ["Q600", "dVa", "dVb", "I0", "I300", "I600"]
W_Q = ["V0", "V600", "Q600", "dVa", "dVb"]
BINS = [-1e-9, 4, 7, 14, 28, 56, np.inf]
BLAB = ["0-4", "4-7", "7-14", "14-28", "28-56", ">56"]


def design(df, temp=False):
    X = df[W_LIN + (["Tmean"] if temp else [])].values.astype(float)
    Z = df[W_Q + (["Tmean"] if temp else [])].values.astype(float)
    iu = np.triu_indices(Z.shape[1])
    return np.hstack([X, (Z[:, :, None] * Z[:, None, :])[:, iu[0], iu[1]]])


class Stats:
    def __init__(self, X, y, g):
        self.X, self.y, self.g = X, y, g
        self.groups = np.unique(g)
        self.s = {k: self._sums(X[g == k], y[g == k]) for k in self.groups}

    @staticmethod
    def _sums(X, y):
        return dict(n=float(len(X)), sx=X.sum(0), sxx=X.T @ X, sy=y.sum(), sxy=X.T @ y)

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
        err = np.zeros(len(ALPHAS))
        for k in groups:
            t = self.total([j for j in groups if j != k])
            Xk, yk = self.X[self.g == k], self.y[self.g == k]
            for i, a in enumerate(ALPHAS):
                err[i] += np.abs(self.solve(t, a)(Xk) - yk).mean()
        return ALPHAS[int(np.argmin(err))]


def uconn():
    W = ROOT / "tables/C10_UCONN_2026-10-23_v1"
    cyc = pd.read_csv(W / "uconn_cycling_table.csv.gz")
    rpt = pd.read_csv(W / "uconn_rpt_table.csv.gz")
    cyc["t"] = pd.to_datetime(cyc.date, format="%Y.%m.%d %H.%M.%S", errors="coerce")
    if cyc.t.isna().mean() > 0.5:
        cyc["t"] = pd.to_datetime(cyc.date, errors="coerce")
    rpt["t"] = pd.to_datetime(rpt.date_start, errors="coerce")
    cyc, rpt = cyc.dropna(subset=["t"]).sort_values("t"), rpt.dropna(subset=["t"]).sort_values("t")
    nxt = pd.merge_asof(cyc, rpt[["cell", "t", "Q_ref"]].rename(columns={"t": "t_next", "Q_ref": "Q"}),
                        left_on="t", right_on="t_next", by="cell", direction="forward")
    prv = pd.merge_asof(cyc[["cell", "t"]], rpt[["cell", "t", "Q_ref"]].rename(columns={"t": "t_prev", "Q_ref": "Q_prev"}),
                        left_on="t", right_on="t_prev", by="cell", direction="backward")
    d = nxt.assign(t_prev=prv.t_prev.values, Q_prev=prv.Q_prev.values).dropna(subset=["Q", "Q_prev"] + W_LIN[:11] + ["Q600", "I0", "I300", "I600"])
    d = d.drop_duplicates(["cell", "t"]).copy()
    s = pd.read_csv(R / "C11_UCONN_LOGO_2026-10-23_v1/scored_logo.csv.gz", parse_dates=["t"])
    grp = s.groupby("cell").group.first()
    d["group"] = d.cell.map(grp)
    d = d.dropna(subset=["group"])
    d["group"] = d.group.astype(int)
    d["dVa"], d["dVb"] = d.V60 - d.V0, d.V600 - d.V300
    d["lag"] = (d.t_next - d.t).dt.total_seconds() / 86400
    d["frac"] = ((d.t - d.t_prev) / (d.t_next - d.t_prev)).astype(float)
    tr = s[["cell", "t", "MW_quad", "M0"]].merge(d, on=["cell", "t"], how="inner")
    assert len(tr) == len(s), (len(tr), len(s))
    S = Stats(design(tr), tr.Q.values, tr.group.values)
    d["pred"], d["M0"] = np.nan, np.nan
    chk = []
    for k in S.groups:
        gs = [j for j in S.groups if j != k]
        f = S.solve(S.total(gs), S.pick_alpha(gs))
        m = (d.group == k).values
        d.loc[m, "pred"] = f(design(d[m]))
        d.loc[m, "M0"] = tr.Q[tr.group != k].mean()
        mt = (tr.group == k).values
        chk.append(np.abs(f(design(tr[mt])) - tr.MW_quad.values[mt]).max())
        chk.append(abs(tr.Q[tr.group != k].mean() - tr.M0.values[mt][0]))
    d["near"] = d.set_index(["cell", "t"]).index.isin(tr.set_index(["cell", "t"]).index)
    return d, float(max(chk))


def kit():
    W = ROOT / "tables/C31_KIT_2026-10-24_v1"
    w = pd.read_csv(W / "kit_windows.csv.gz").sort_values("t")
    lab = pd.read_csv(W / "kit_checkup_labels.csv").sort_values("t_lab")
    nx = pd.merge_asof(w, lab.rename(columns={"t_lab": "t_next"}), left_on="t", right_on="t_next", by="cell", direction="forward")
    pv = pd.merge_asof(w[["cell", "t"]], lab.rename(columns={"t_lab": "t_prev", "Q": "Q_prev"}), left_on="t", right_on="t_prev",
                       by="cell", direction="backward")
    d = nx.assign(t_prev=pv.t_prev.values, Q_prev=pv.Q_prev.values).dropna(subset=["Q", "Q_prev"]).copy()
    d["dVa"], d["dVb"] = d.V60 - d.V0, d.V600 - d.V300
    d["Tmean"] = d[[f"T{s}" for s in range(0, 601, 30)]].mean(1)
    d = d.dropna(subset=W_LIN + ["Tmean"])
    d["lag"] = (d.t_next - d.t) / 86400
    d["frac"] = (d.t - d.t_prev) / (d.t_next - d.t_prev)
    s = pd.read_csv(R / "C31_KIT_LOGO_2026-10-24_usage/scored.csv.gz")
    d["group"] = d.cell.map(s.groupby("cell").group.first())
    d = d.dropna(subset=["group"])
    d["group"] = d.group.astype(int)
    tr = s[["cell", "t", "MW_quadT", "M0"]].merge(d, on=["cell", "t"], how="inner")
    assert len(tr) == len(s), (len(tr), len(s))
    S = Stats(design(tr, True), tr.Q.values, tr.group.values)
    d["pred"], d["M0"] = np.nan, np.nan
    chk = []
    for k in S.groups:
        gs = [j for j in S.groups if j != k]
        f = S.solve(S.total(gs), S.pick_alpha(gs))
        m = (d.group == k).values
        d.loc[m, "pred"] = f(design(d[m], True))
        d.loc[m, "M0"] = tr.Q[tr.group != k].mean()
        mt = (tr.group == k).values
        chk.append(np.abs(f(design(tr[mt], True)) - tr.MW_quadT.values[mt]).max())
        chk.append(abs(tr.Q[tr.group != k].mean() - tr.M0.values[mt][0]))
    d["near"] = d.set_index(["cell", "t"]).index.isin(tr.set_index(["cell", "t"]).index)
    return d, float(max(chk))


def eq(x, col, lab):
    """mean over groups of equal-cell MAE; also signed bias (equal-cell)"""
    e = x.assign(ae=(x[col] - x[lab]).abs(), se=x[col] - x[lab])
    c = e.groupby(["group", "cell"])[["ae", "se"]].mean().groupby("group").mean()
    return c


def eq_iv(x, col, lab):
    e = x.assign(ae=(x[col] - x[lab]).abs())
    return e.groupby(["group", "cell", "t_next"]).ae.mean().groupby(["group", "cell"]).mean().groupby("group").mean().mean()


def table(d, lab):
    rows = []
    d = d.assign(bin=pd.cut(d.lag, BINS, labels=BLAB))
    for b, x in [("near", d[d.near]), ("all", d)] + [(b, d[d.bin == b]) for b in BLAB]:
        if len(x) == 0:
            continue
        s, m = eq(x, "pred", lab), eq(x, "M0", lab)
        rows.append(dict(bin=b, rows=len(x), cells=x.cell.nunique(), groups=len(s), mean_lag_d=round(x.lag.mean(), 2),
                         sensor_MAE=round(s.ae.mean(), 4), M0_MAE=round(m.ae.mean(), 4), sensor_bias=round(s.se.mean(), 4),
                         groups_sensor_beats_M0=int((s.ae < m.ae).sum()),
                         sensor_MAE_interval_weighted=round(float(eq_iv(x, "pred", lab)), 4)))
    return pd.DataFrame(rows)


res = {}
for name, fn in (("UConn", uconn), ("KIT", kit)):
    d, chk = fn()
    d["Q_interp"] = d.Q_prev + (d.Q - d.Q_prev) * d.frac
    res[name] = dict(sanity=chk)
    lim = 4 if name == "UConn" else 7
    pool = d[d.lag <= lim]
    draws = [float(eq(pool.sample(frac=1.0, random_state=r).groupby(["cell", "t_next"]).head(10), "pred", "Q").ae.mean())
             for r in range(2, 22)]
    res[name]["near_resample_seeds2_21"] = dict(min=round(min(draws), 4), median=round(float(np.median(draws)), 4),
                                                max=round(max(draws), 4), locked=round(float(eq(d[d.near], "pred", "Q").ae.mean()), 4))
    for lab in ("Q", "Q_interp"):
        t = table(d, lab)
        t.to_csv(OUT / f"{name}_{lab}_by_lag.csv", index=False)
        bins = t[t.bin.isin(BLAB)]
        res[name][lab] = dict(
            table=t.to_dict("records"),
            L1_sensor_beats_M0_every_bin=bool((bins.sensor_MAE < bins.M0_MAE).all()),
            L2_all_over_near=round(float(t.set_index("bin").loc["all", "sensor_MAE"] / t.set_index("bin").loc["near", "sensor_MAE"]), 3))
        print(name, lab, "\n", t.to_string(index=False), flush=True)
    d[["cell", "group", "t", "t_next", "lag", "near", "Q", "Q_prev", "Q_interp", "pred", "M0"]].to_csv(OUT / f"{name}_rows.csv.gz", index=False)
res["UConn"]["sanity_pass"] = res["UConn"]["sanity"] <= 1e-6
res["KIT"]["sanity_pass"] = res["KIT"]["sanity"] <= 1e-6
json.dump(res, open(OUT / "summary.json", "w"), indent=2, default=str)
print(json.dumps({n: {"sanity": res[n]["sanity"], "pass": res[n]["sanity_pass"]} for n in res}, indent=1, default=str))
