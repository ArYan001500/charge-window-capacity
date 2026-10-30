"""KIT: charge-irregularity term in the fade-rate model."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=False)
W = ROOT / "tables/C31_KIT_2026-10-24_v1"
w = pd.read_csv(W / "kit_windows.csv.gz", usecols=["cell", "t", "dur", "v_pre", "chg", "dis", "vmax", "vmin", "temp", "I0", "I300", "I600"])
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
                         n=int(m.sum()), ah=float(x.ah[m].sum()), hrs=float(x.hrs[m].sum()),
                         sd_vpre=float(x.v_pre[m].std()) if m.sum() > 2 else 0.0, sd_dur=float(x.hrs[m].std()) if m.sum() > 2 else 0.0,
                         cv_gap=float(np.diff(x.t[m].values).std() / max(np.diff(x.t[m].values).mean(), 1.0)) if m.sum() > 3 else 0.0,
                         **cfg.loc[c].to_dict()))
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
BASE_TERMS = dict(TERMS)
IRR = {"sd_vpre": lambda d: d.sd_vpre, "sd_dur": lambda d: d.sd_dur, "cv_gap": lambda d: d.cv_gap,
       "ahd_sd_vpre": lambda d: d.ahd * d.sd_vpre}


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


res, preds = {}, {}
for name, terms in (("M6", BASE_TERMS), ("M6_irr", {**BASE_TERMS, **IRR})):
    TERMS.clear(); TERMS.update(terms)
    P, forms = [], {}
    for g in sorted(E.group.unique()):
        S, D = E[E.group != g], E[E.group == g].copy()
        ch = stepwise(S)
        forms[int(g)] = ch
        D["pred"] = lin_fit(S, ch)(D)
        P.append(D)
    P = pd.concat(P)
    preds[name] = P
    pg = P.groupby("group").apply(lambda z: eqmae(z, z.pred))
    res[name] = dict(mae=round(float(pg.mean()), 4), formulas=forms)
a, b = preds["M6_irr"], preds["M6"]
T = pd.DataFrame(dict(group=a.group.values, cell=a.cell.values, a=(a.Q0 + a.pred * a.dt - a.Q1).abs().values,
                      b=(b.Q0 + b.pred * b.dt - b.Q1).abs().values)).groupby(["group", "cell"])[["a", "b"]].mean().reset_index()
rng = np.random.default_rng(68)
diffs = [np.mean([(lambda z: (z.a - z.b).mean())(x.sample(len(x), replace=True, random_state=int(rng.integers(1e9)))) for _, x in T.groupby("group")]) for _ in range(2000)]
ci = np.percentile(diffs, [2.5, 97.5])
sel = sum(any(k in IRR for k in f) for f in res["M6_irr"]["formulas"].values())
res.update(diff_CI95=[round(float(ci[0]), 4), round(float(ci[1]), 4)], folds_with_irregularity_term=f"{sel}/13",
           K_pass=bool(ci[1] < 0 and sel >= 7),
           irregularity_spread=E[["sd_vpre", "sd_dur", "cv_gap"]].describe().round(3).to_dict())
json.dump(res, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps({k: v for k, v in res.items() if k != "irregularity_spread"}, indent=1))
