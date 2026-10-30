"""KIT: fade-rate model inside the filter."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import c57_kf_intervals as K

OUT = Path(sys.argv[2])
d, ev, R_TEST, slopes, _ = K.DATA["KIT"]
W = K.ROOT / "tables/C31_KIT_2026-10-24_v1"
w = pd.read_csv(W / "kit_windows.csv.gz", usecols=["cell", "t", "dur", "chg", "dis", "vmax", "vmin", "temp", "I0", "I300", "I600"])
w["ah"] = w[["I0", "I300", "I600"]].abs().mean(1) * w.dur / 3600
w["hrs"] = w.dur / 3600
w["day"] = w.t / 86400
w = w.sort_values(["cell", "day"])
CFG = w.groupby("cell")[["chg", "dis", "vmax", "vmin", "temp"]].first()
CFGA = {c: r.values.astype(float) for c, r in CFG.iterrows()}
CUM = {c: (x.day.values, np.cumsum(x.ah.values), np.cumsum(x.hrs.values), np.arange(1, len(x) + 1)) for c, x in w.groupby("cell")}
lab = pd.read_csv(W / "kit_checkup_labels.csv").sort_values(["cell", "t_lab"])
cg = d.groupby("cell").group.first()


def usage(c, a, b):
    t, ca, ch, cn = CUM[c]
    i, j = np.searchsorted(t, a, "right"), np.searchsorted(t, b, "right")
    get = lambda arr, k: arr[k - 1] if k > 0 else 0.0
    return get(ca, j) - get(ca, i), get(ch, j) - get(ch, i), get(cn, j) - get(cn, i)


rows = []
for c, L in lab.groupby("cell"):
    if c not in cg.index:
        continue
    t, q = L.t_lab.values / 86400, L.Q.values
    pr = np.nan
    for j in range(len(t) - 1):
        dt = t[j + 1] - t[j]
        if dt <= 0.5:
            continue
        ah, hr, n = usage(c, t[j], t[j + 1])
        cf = CFG.loc[c]
        rows.append(dict(cell=c, group=int(cg[c]), dt=dt, Q0=q[j], Q1=q[j + 1], rate=(q[j + 1] - q[j]) / dt, prev=pr,
                         ahd=ah / dt, hd=hr / dt, nd=n / dt, T=cf.temp, chg=cf.chg, dv=cf.vmax - cf.vmin))
        pr = (q[j + 1] - q[j]) / dt
IV = pd.DataFrame(rows)

USE = {"D2": ["ahd"], "D3": ["hd", "nd"], "D4": ["ahd", "hd", "nd"]}


def terms_for(use, own):
    T = {}
    for u in use:
        T[u] = lambda X, u=u: X[u]
        T[f"{u}_chg"] = lambda X, u=u: X[u] * X.chg
        T[f"{u}_dv"] = lambda X, u=u: X[u] * X.dv
        T[f"{u}_T"] = lambda X, u=u: X[u] * X["T"]
    T["T"] = lambda X: X["T"]
    if own:
        T["prev"] = lambda X: X.prev
        T["Q0"] = lambda X: X.Q0
    return T


class Row(dict):
    __getattr__ = dict.__getitem__


def fit_lin(S, TT, chosen):
    if not chosen:
        m = S.rate.mean()
        f = lambda X: np.full(len(X), m)
        f.row = lambda r: m
        return f
    A = np.column_stack([TT[k](S) for k in chosen] + [np.ones(len(S))])
    coef = np.linalg.lstsq(A, S.rate.values, rcond=None)[0]
    f = lambda X: np.column_stack([TT[k](X) for k in chosen] + [np.ones(len(X))]) @ coef
    f.row = lambda r: float(sum(cf * TT[k](r) for cf, k in zip(coef[:-1], chosen)) + coef[-1])
    return f


def mae_iv(D, f):
    return float((D.Q0 + f(D) * D.dt - D.Q1).abs().groupby(D.cell).mean().mean())


def stepwise(S, TT):
    gs = S.group.unique()
    score = lambda ch: np.mean([mae_iv(S[S.group == h], fit_lin(S[S.group != h], TT, ch)) for h in gs])
    chosen, best = [], score([])
    while True:
        c = [(score(chosen + [k]), k) for k in TT if k not in chosen]
        if not c:
            break
        m, k = min(c)
        if m >= best:
            break
        chosen, best = chosen + [k], m
    return chosen, fit_lin(S, TT, chosen)


def make_drift(model, g):
    S = IV[IV.group != g]
    d0 = float(np.mean([slopes[c] for c in cg[cg != g].index if c in slopes]))
    if model == "D0":
        return (lambda c, st: d0), {}
    if model == "D1":
        return (lambda c, st: st["prev"] if st["prev"] is not None else d0), {}
    use = USE[model]
    TT0, TT1 = terms_for(use, False), terms_for(use, True)
    ch0, f0 = stepwise(S, TT0)
    S1 = S.dropna(subset=["prev"])
    ch1, f1 = stepwise(S1, TT1)

    def drift(c, st):
        if st["now"] - st["tL"] >= 7 or st["tP"] is None:
            a, b = (st["tL"], st["now"]) if st["now"] - st["tL"] >= 7 else (st["now"] - 7, st["now"])
        else:
            a, b = st["tP"], st["tL"]
        span = max(b - a, 1.0)
        ah, hr, n = usage(c, a, b)
        cf = CFGA[c]
        X = Row(ahd=ah / span, hd=hr / span, nd=n / span, T=cf[4], chg=cf[0], dv=cf[2] - cf[3],
                prev=st["prev"] if st["prev"] is not None else np.nan, Q0=st["Q0"])
        return (f1 if st["prev"] is not None else f0).row(X)
    return drift, {"formula_first_interval": ch0, "formula_with_history": ch1}


def kf(c, drift, qn, qb, r_w, p_b0, use_w):
    s, out, prev = None, {}, None
    st = dict(prev=None, tL=None, tP=None, Q0=None, now=None)
    rate = 0.0
    for day, is_test, z, row in ev[c]:
        if s is not None:
            dt = day - prev
            s = s + np.array([rate * dt, 0.0])
            P = P + np.diag([qn * dt, qb * dt])
        prev = day
        if is_test:
            if s is None:
                s, P = np.array([z, 0.0]), np.diag([R_TEST, p_b0])
            else:
                H = np.array([1.0, 0.0])
                Kg = P @ H / (H @ P @ H + R_TEST)
                s, P = s + Kg * (z - H @ s), P - np.outer(Kg, H @ P)
                if st["tL"] is not None and day - st["tL"] > 0.5:
                    st["prev"] = (z - st["Q0"]) / (day - st["tL"])
            st["tP"], st["tL"], st["Q0"] = st["tL"], day, z
        elif s is not None:
            if use_w:
                H = np.array([1.0, 1.0])
                Kg = P @ H / (H @ P @ H + r_w)
                s, P = s + Kg * (z - H @ s), P - np.outer(Kg, H @ P)
            out[row] = s[0]
        if s is not None:
            st["now"] = day
            rate = drift(c, st)
    return out


def eqmae(o):
    x = d.loc[[k for k in o if d.at[k, "score_row"]]]
    return float((pd.Series({k: o[k] for k in x.index}) - x.Q).abs().groupby(x.cell).mean().mean()), x


def job(args):
    g, model, use_w = args
    drift, info = make_drift(model, g)
    src = list(cg[cg != g].index)
    s = d[d.cell.isin(src) & d.score_row]
    bias = (s.pred - s.Q).groupby(s.cell).mean()
    p_b0 = float(np.var(bias) + np.mean(bias) ** 2)
    r_w = float(np.var((s.pred - s.Q) - s.cell.map(bias)))
    grid = K.GRID if use_w else [(qn, 1e-6) for qn in (1e-5, 1e-4, 1e-3)]

    def run(cells, p):
        o = {}
        for c in cells:
            o.update(kf(c, drift, p[0], p[1], r_w, p_b0, use_w))
        return o
    best = min(grid, key=lambda p: eqmae(run(src, p))[0])
    o = run(list(cg[cg == g].index), best)
    m, x = eqmae(o)
    pc = (pd.Series({k: o[k] for k in x.index}) - x.Q).abs().groupby(x.cell).mean().to_dict()
    return dict(group=int(g), model=model, windows=use_w, mae=m, cells=pc, **{k: str(v) for k, v in info.items()})


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=False)
    jobs = [(g, m, uw) for g in sorted(cg.unique()) for m in ("D0", "D1", "D2", "D3", "D4") for uw in (True, False)]
    with Pool(18) as pool:
        r = pd.DataFrame(pool.map(job, jobs))
    r.drop(columns="cells").to_csv(OUT / "per_group.csv", index=False)
    P = r.pivot_table(index="group", columns=["model", "windows"], values="mae")
    res = {f"{m}_{'win' if uw else 'nowin'}": round(float(P[(m, uw)].mean()), 4) for m, uw in P.columns}
    fast = (6, 7, 10, 11)
    best = min(("D2", "D3", "D4"), key=lambda m: P[(m, True)].mean())
    res["best_formula"] = best
    res["fast_groups"] = {int(g): {k: round(float(P.loc[g, (k, True)]), 4) for k in ("D0", "D1", "D2", "D3", "D4")} for g in fast}
    res["K2_fast_improved"] = f"{int(sum(P.loc[g, (best, True)] < P.loc[g, ('D0', True)] for g in fast))}/4"
    res["group_wins_best_vs_D0"] = f"{int((P[(best, True)] < P[('D0', True)]).sum())}/{len(P)}"
    def cells(m, uw):
        return {(rr.group, c): v for rr in r[(r.model == m) & (r.windows == uw)].itertuples() for c, v in rr.cells.items()}
    a, b = cells(best, True), cells("D0", True)
    T = pd.DataFrame([dict(group=k[0], a=a[k], b=b[k]) for k in a if k in b])
    rng = np.random.default_rng(65)
    diffs = [np.mean([(lambda z: (z.a - z.b).mean())(x.sample(len(x), replace=True, random_state=int(rng.integers(1e9)))) for _, x in T.groupby("group")])
             for _ in range(2000)]
    ci = np.percentile(diffs, [2.5, 97.5])
    res["best_minus_D0_CI95"] = [round(float(ci[0]), 4), round(float(ci[1]), 4)]
    res["K1_pass"] = bool(ci[1] < 0)
    res["formulas"] = r[(r.model.isin(["D2", "D3", "D4"])) & r.windows][["group", "model", "formula_with_history"]].to_dict("records")
    res["H_hours_D3_over_D2"] = round(float(P[("D3", True)].mean() / P[("D2", True)].mean()), 3)
    json.dump(res, open(OUT / "summary.json", "w"), indent=2)
    print(json.dumps(res, indent=1))
