"""KIT: Kalman fusion of window estimates with check-up capacities (first implementation)."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

W_DIR, OUT = Path(sys.argv[1]), Path(sys.argv[2])
PRED = Path(sys.argv[3]) if len(sys.argv) > 3 else None
VG = [f"V{s}" for s in range(0, 601, 60)]
FL = VG + ["Q600", "dVa", "dVb", "I0", "I300", "I600", "Tmean"]
FQ = ["V0", "V600", "Q600", "dVa", "dVb", "Tmean"]


def load():
    w = pd.read_csv(W_DIR / "kit_windows.csv.gz").sort_values("t")
    lab = pd.read_csv(W_DIR / "kit_checkup_labels.csv").sort_values("t_lab")
    d = pd.merge_asof(w, lab.rename(columns={"t_lab": "t_next"}), left_on="t", right_on="t_next", by="cell", direction="forward")
    prv = pd.merge_asof(w[["cell", "t"]], lab.rename(columns={"t_lab": "t_prev", "Q": "Q_prev"}), left_on="t", right_on="t_prev",
                        by="cell", direction="backward")
    d = d.assign(t_prev=prv.t_prev.values, Q_prev=prv.Q_prev.values).dropna(subset=["Q", "Q_prev"]).copy()
    d["near"] = (d.t_next - d.t) <= 7 * 86400
    d = d.sample(frac=1.0, random_state=1)
    d = pd.concat([d[d.near].groupby(["cell", "t_next"]).head(10), d[~d.near].groupby(["cell", "t_next"]).head(20)])
    combo = d.apply(lambda r: "WLTP" if r.age_type == 3 else f"{r.vmax}_{r.vmin}_{r.chg}_{r.dis}", axis=1)
    d["group"] = combo.astype("category").cat.codes
    d["dVa"], d["dVb"] = d.V60 - d.V0, d.V600 - d.V300
    d["Tmean"] = d[[f"T{s}" for s in range(0, 601, 30)]].mean(1)
    return d.sort_values(["cell", "t"]).reset_index(drop=True), lab


def design(df):
    X = df[FL].values.astype(float)
    Z = df[FQ].values.astype(float)
    iu = np.triu_indices(Z.shape[1])
    return np.hstack([X, (Z[:, :, None] * Z[:, None, :])[:, iu[0], iu[1]]])


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Zs = (X - mu) / sd
    w = np.linalg.solve(Zs.T @ Zs + alpha * np.eye(Zs.shape[1]), Zs.T @ (y - y.mean()))
    return lambda Xn: (Xn - mu) / sd @ w + y.mean()


def pick_alpha(X, y, g):
    return min(np.logspace(-3, 4, 15), key=lambda a: np.mean(
        [np.abs(ridge_fit(X[g != k], y[g != k], a)(X[g == k]) - y[g == k]).mean() for k in np.unique(g)]))


def predict_group(args):
    g, d = args
    src = d[(d.group != g) & d.near]
    Xs = design(src)
    f = ridge_fit(Xs, src.Q.values, pick_alpha(Xs, src.Q.values, src.group.values))
    te = d[d.group == g]
    return pd.Series(f(design(te)), index=te.index)


def run_cell(x, labs, k_every, drift, qn, qb, r_test, r_win, p_b0, use_w):
    """x: windows of one cell (time order); labs: check-ups of that cell (time order)."""
    tests = labs.iloc[::k_every]
    ev = sorted([(t, "test", q) for t, q in zip(tests.t_lab, tests.Q)] +
                [(t, "win", i) for t, i in zip(x.t, x.index)], key=lambda e: e[0])
    s = None
    out = {}
    prev = None
    for t, kind, val in ev:
        if s is not None:
            dt = (t - prev) / 86400
            s = s + np.array([drift * dt, 0.0])
            P = P + np.diag([qn * dt, qb * dt])
        prev = t
        if kind == "test":
            if s is None:
                s, P = np.array([val, 0.0]), np.diag([r_test, p_b0])
            else:
                H = np.array([1.0, 0.0])
                K = P @ H / (H @ P @ H + r_test)
                s, P = s + K * (val - H @ s), P - np.outer(K, H @ P)
        else:
            if s is None:
                continue
            if use_w:
                H = np.array([1.0, 1.0])
                K = P @ H / (H @ P @ H + r_win)
                s, P = s + K * (x.at[val, "pred"] - H @ s), P - np.outer(K, H @ P)
            out[val] = s[0]
    return out


def evaluate(cells, d, labs_by_cell, k_every, params, use_w):
    est = {}
    for c in cells:
        x = d[d.cell == c]
        est.update(run_cell(x, labs_by_cell[c], k_every, *params, use_w))
    e = d.loc[list(est)].assign(est=pd.Series(est))
    e = e[e.near]
    return e


def score(e):
    return e.groupby("group").apply(lambda z: (z.est - z.Q).abs().groupby(z.cell).mean().mean()).mean()


def job(args):
    g, k_every, d, labs_by_cell, cells_by_group = args
    src_cells = [c for gg, cs in cells_by_group.items() if gg != g for c in cs]
    s = d[d.cell.isin(src_cells) & d.near]
    slope = []
    for c in src_cells:
        L = labs_by_cell[c]
        if len(L) >= 2:
            slope.append(np.polyfit(L.t_lab / 86400, L.Q, 1)[0])
    drift = float(np.mean(slope))
    r_test = 1e-4
    bias = (s.pred - s.Q).groupby(s.cell).mean()
    p_b0 = float(np.var(bias) + np.mean(bias) ** 2)
    r_win = float(np.var((s.pred - s.Q) - s.cell.map(bias)))
    res = {}
    for name, use_w in (("ref_drift", False), ("bias_kf", True)):
        best, bp = np.inf, None
        for qn in (1e-5, 1e-4, 1e-3):
            for qb in ((1e-5, 1e-4, 1e-3) if use_w else (1e-5,)):
                m = score(evaluate(src_cells, d, labs_by_cell, k_every, (drift, qn, qb, r_test, r_win, p_b0), use_w))
                if m < best:
                    best, bp = m, (qn, qb)
        e = evaluate(cells_by_group[g], d, labs_by_cell, k_every, (drift, bp[0], bp[1], r_test, r_win, p_b0), use_w)
        res[name] = float((e.est - e.Q).abs().groupby(e.cell).mean().mean())
    return dict(group=g, k_every=k_every, **res)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=False)
    d, lab = load()
    if PRED is None:
        with Pool(13) as pool:
            preds = pool.map(predict_group, [(g, d) for g in sorted(d.group.unique())])
        d["pred"] = pd.concat(preds)
    else:
        pp = pd.read_csv(PRED / "window_predictions.csv.gz")[["cell", "t", "pred"]]
        d = d.merge(pp, on=["cell", "t"], how="inner").sort_values(["cell", "t"]).reset_index(drop=True)
    d[["cell", "group", "t", "near", "Q", "pred"]].to_csv(OUT / "window_predictions.csv.gz", index=False)
    labs_by_cell = {c: lab[lab.cell == c].reset_index(drop=True) for c in d.cell.unique()}
    cells_by_group = {g: list(x.cell.unique()) for g, x in d.groupby("group")}
    passive = d[d.near].groupby("group").apply(lambda z: (z.pred - z.Q).abs().groupby(z.cell).mean().mean())
    jobs = [(g, k, d, labs_by_cell, cells_by_group) for k in (1, 2) for g in sorted(cells_by_group)]
    with Pool(18) as pool:
        rows = pool.map(job, jobs)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "per_group.csv", index=False)
    summ = {f"k{k}": dict(ref_drift=float(x.ref_drift.mean()), bias_kf=float(x.bias_kf.mean()),
                          kf_wins=int((x.bias_kf < x.ref_drift).sum()), n_groups=len(x)) for k, x in r.groupby("k_every")}
    summ["passive_only_MW_quadT"] = float(passive.mean())
    json.dump(summ, open(OUT / "summary.json", "w"), indent=2)
    print(json.dumps(summ, indent=1))
