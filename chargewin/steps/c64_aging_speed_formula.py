"""KIT: fade-rate model for the next check-up."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=False)
W = ROOT / "tables/C31_KIT_2026-10-24_v1"
w = pd.read_csv(W / "kit_windows.csv.gz", usecols=["cell", "t", "dur", "chg", "dis", "vmax", "vmin", "temp", "I0", "I300", "I600"])
lab = pd.read_csv(W / "kit_checkup_labels.csv").sort_values(["cell", "t_lab"])
grp = pd.read_csv(ROOT / "results/C31_KIT_LOGO_2026-10-24_usage/scored.csv.gz", usecols=["cell", "group"]).drop_duplicates().set_index("cell").group
w["ah"] = w[["I0", "I300", "I600"]].abs().mean(1) * w.dur / 3600
w["hrs"] = w.dur / 3600
cfg = w.groupby("cell")[["chg", "dis", "vmax", "vmin", "temp"]].first()

rows = []
for c, L in lab.groupby("cell"):
    if c not in grp.index:
        continue
    t, q = L.t_lab.values, L.Q.values
    x = w[w.cell == c]
    for j in range(len(t) - 1):
        m = (x.t > t[j]) & (x.t <= t[j + 1])
        rows.append(dict(cell=c, group=int(grp[c]), j=j, dt=(t[j + 1] - t[j]) / 86400, Q0=q[j], Q1=q[j + 1], dQ=q[j + 1] - q[j],
                         n=int(m.sum()), ah=float(x.ah[m].sum()), hrs=float(x.hrs[m].sum()), **cfg.loc[c].to_dict()))
I = pd.DataFrame(rows)
I = I[I.dt > 0.5].copy()
I["rate_d"], I["rate_ah"] = I.dQ / I.dt, I.dQ / I.ah.clip(lower=1e-3)
I["ahd"] = I.ah / I.dt
g1 = I.groupby("cell")
I["prev_rate_d"], I["prev_rate_ah"] = g1.rate_d.shift(1), g1.rate_ah.shift(1)
I["n_seen"] = g1.cumcount()
I["dv"] = I.vmax - I.vmin
I["T"] = I.temp
E = I[I.j >= 1].dropna(subset=["prev_rate_d"]).copy()
E.to_csv(OUT / "intervals.csv", index=False)


def ridge(X, y, a=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu) / sd
    wv = np.linalg.solve(Z.T @ Z + a * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return lambda Xn: (Xn - mu) / sd @ wv + y.mean()


TERMS = {"ahd": lambda d: d.ahd, "ahd_chg": lambda d: d.ahd * d.chg, "ahd_dv": lambda d: d.ahd * d.dv, "T": lambda d: d["T"],
         "ahd_T": lambda d: d.ahd * d["T"], "prev_rate_d": lambda d: d.prev_rate_d, "prev_rate_ah_x_ahd": lambda d: d.prev_rate_ah * d.ahd,
         "Q0": lambda d: d.Q0}


def lin_fit(S, terms):
    if not terms:
        m = S.rate_d.mean()
        return lambda D: np.full(len(D), m)
    X = np.column_stack([TERMS[k](S) for k in terms])
    f = ridge(X, S.rate_d.values)
    return lambda D: f(np.column_stack([TERMS[k](D) for k in terms]))


def eqmae(D, rate):
    return float((D.Q0 + rate * D.dt - D.Q1).abs().groupby(D.cell).mean().mean())


def stepwise(S):
    chosen, best = [], np.mean([eqmae(S[S.group == h], lin_fit(S[S.group != h], [])(S[S.group == h])) for h in S.group.unique()])
    while True:
        cand = []
        for k in TERMS:
            if k in chosen:
                continue
            m = np.mean([eqmae(S[S.group == h], lin_fit(S[S.group != h], chosen + [k])(S[S.group == h])) for h in S.group.unique()])
            cand.append((m, k))
        if not cand:
            return chosen
        m, k = min(cand)
        if m >= best:
            return chosen
        chosen, best = chosen + [k], m


preds, formulas = [], {}
for g in sorted(E.group.unique()):
    S, D = E[E.group != g], E[E.group == g].copy()
    Sall = I[I.group != g]
    r_d = Sall.rate_d.mean()
    r_ah = Sall.dQ.sum() / Sall.ah.sum()
    D["M1"] = r_d
    D["M2"] = r_ah * D.ahd
    D["M3"] = D.prev_rate_d
    D["M4"] = D.prev_rate_ah * D.ahd

    def m5(X, n0):
        wgt = X.n_seen / (X.n_seen + n0)
        return wgt * X.prev_rate_ah * X.ahd + (1 - wgt) * r_ah * X.ahd
    n0 = min((0.5, 1, 2, 4, 8), key=lambda n: eqmae(S, m5(S, n)))
    D["M5"] = m5(D, n0)
    terms = stepwise(S)
    formulas[int(g)] = terms
    D["M6"] = lin_fit(S, terms)(D)
    X7 = lambda X: np.column_stack([np.log(X.ahd.clip(lower=1e-3)), X.chg, X.dis, X.dv, 1 / (X["T"] + 273.15)])
    f7 = ridge(X7(S), np.log((-S.rate_d).clip(lower=1e-5)), a=1.0)
    D["M7"] = -np.exp(f7(X7(D)))
    D["n0"] = n0
    preds.append(D)
P = pd.concat(preds)
P.to_csv(OUT / "predictions.csv", index=False)
M = ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]
pg = pd.DataFrame({m: P.groupby("group").apply(lambda z: eqmae(z, z[m])) for m in M})
res = dict(n_intervals=len(P), cells=int(P.cell.nunique()), mean=pg.mean().round(4).to_dict(),
           fast_groups={int(g): pg.loc[g].round(4).to_dict() for g in (6, 7, 10, 11)}, formulas=formulas)
cand = ["M4", "M5", "M6", "M7"]
best = min(cand, key=lambda m: pg[m].mean())
ae = lambda m: (P.Q0 + P[m] * P.dt - P.Q1).abs()
T = P.assign(a=ae(best), b=ae("M1")).groupby(["group", "cell"])[["a", "b"]].mean().reset_index()
rng = np.random.default_rng(64)
diffs = [np.mean([(lambda z: (z.a - z.b).mean())(x.sample(len(x), replace=True, random_state=int(rng.integers(1e9)))) for _, x in T.groupby("group")])
         for _ in range(2000)]
ci = np.percentile(diffs, [2.5, 97.5])
res.update(best=best, best_minus_M1_CI95=[round(float(ci[0]), 4), round(float(ci[1]), 4)], K1_pass=bool(ci[1] < 0),
           K2_fast_improved=f"{int(sum(pg.loc[g, best] < pg.loc[g, 'M1'] for g in (6, 7, 10, 11)))}/4",
           group_wins_vs_M1=f"{int((pg[best] < pg.M1).sum())}/{len(pg)}")
res["K2_pass"] = int(res["K2_fast_improved"][0]) >= 3
json.dump(res, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(res, indent=1))
