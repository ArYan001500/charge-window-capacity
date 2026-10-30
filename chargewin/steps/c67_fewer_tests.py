"""KIT: variance-based trigger that requests fewer tests."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import c65_formula_drift_kalman as C

OUT = Path(sys.argv[2])
d, ev, cg = C.d, C.ev, C.cg
w = pd.read_csv(C.W / "kit_windows.csv.gz", usecols=["cell", "t", "V0", "chg"])
dd = d[["cell", "t", "group"]].merge(w, on=["cell", "t"], how="left")
zs = np.zeros(len(d), bool)
for g in d.group.unique():
    tgt = np.where(dd.group.values == g)[0]
    cnt = np.zeros(len(tgt), int)
    for h in d.group.unique():
        if h == g:
            continue
        s = dd[dd.group == h]
        n = np.zeros(len(tgt), int)
        for c in np.unique(dd.chg.values[tgt]):
            v = np.sort(s.V0.values[s.chg.values == c])
            m = dd.chg.values[tgt] == c
            x = dd.V0.values[tgt][m]
            n[m] = np.searchsorted(v, x + 0.03) - np.searchsorted(v, x - 0.03)
        cnt += n >= 20
    zs[tgt] = cnt == 0
ZS = zs
TAUS = [0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30]


def kf(c, drift, qn, qb, r_w, p_b0, policy, rng):
    s, out, prev = None, {}, None
    st = dict(prev=None, tL=None, tP=None, Q0=None, now=None)
    rate, used, avail, since = 0.0, 0, 0, []
    for day, is_test, z, row in ev[c]:
        if s is not None:
            dt = day - prev
            s = s + np.array([rate * dt, 0.0])
            P = P + np.diag([qn * dt, qb * dt])
        prev = day
        if is_test:
            avail += 1
            take = s is None or policy(np.sqrt(P[0, 0]), np.mean(since) if since else 0.0, avail, rng)
            if take:
                used += 1
                since = []
                if s is None:
                    s, P = np.array([z, 0.0]), np.diag([C.R_TEST, p_b0])
                else:
                    H = np.array([1.0, 0.0])
                    Kg = P @ H / (H @ P @ H + C.R_TEST)
                    s, P = s + Kg * (z - H @ s), P - np.outer(Kg, H @ P)
                    if st["tL"] is not None and day - st["tL"] > 0.5:
                        st["prev"] = (z - st["Q0"]) / (day - st["tL"])
                st["tP"], st["tL"], st["Q0"] = st["tL"], day, z
        elif s is not None:
            H = np.array([1.0, 1.0])
            Kg = P @ H / (H @ P @ H + r_w)
            s, P = s + Kg * (z - H @ s), P - np.outer(Kg, H @ P)
            out[row] = s[0]
            since.append(ZS[row])
        if s is not None:
            st["now"] = day
            rate = drift(c, st)
    return out, used, avail


def run(cells, drift, prm, policy, seed=0):
    rng = np.random.default_rng(seed)
    o, u, a = {}, 0, 0
    for c in cells:
        oo, uu, aa = kf(c, drift, *prm, policy, rng)
        o.update(oo)
        u, a = u + uu, a + aa
    m, x = C.eqmae(o)
    pc = (pd.Series({k: o[k] for k in x.index}) - x.Q).abs().groupby(x.cell).mean().to_dict()
    return m, u / a, pc


def job(g):
    drift, _ = C.make_drift("D1", g)
    src, tgt = list(cg[cg != g].index), list(cg[cg == g].index)
    s = d[d.cell.isin(src) & d.score_row]
    bias = (s.pred - s.Q).groupby(s.cell).mean()
    p_b0 = float(np.var(bias) + np.mean(bias) ** 2)
    r_w = float(np.var((s.pred - s.Q) - s.cell.map(bias)))
    allp = lambda sd, f, a, r: True
    best = min(C.K.GRID, key=lambda p: run(src, drift, (p[0], p[1], r_w, p_b0), allp)[0])
    prm = (best[0], best[1], r_w, p_b0)
    pols = {"ALL": allp, "PER2": lambda sd, f, a, r: (a - 1) % 2 == 0, "PER3": lambda sd, f, a, r: (a - 1) % 3 == 0}
    out = []
    for name, pol in pols.items():
        m, b, pc = run(tgt, drift, prm, pol)
        out.append(dict(group=int(g), policy=name, mae=m, budget=b, cells=pc))
    for p in (0.5, 0.33):
        res = [run(tgt, drift, prm, (lambda p: lambda sd, f, a, r: r.random() < p)(p), seed=k) for k in range(5)]
        pc = {c: np.mean([r[2][c] for r in res]) for c in res[0][2]}
        out.append(dict(group=int(g), policy=f"RND{p}", mae=float(np.mean([r[0] for r in res])), budget=float(np.mean([r[1] for r in res])), cells=pc))
    src_budget = {}
    for tau in TAUS:
        for sup in (False, True):
            pol = (lambda t, sp: lambda sd, f, a, r: sd > t or (sp and f > 0.3))(tau, sup)
            src_budget[(tau, sup)] = run(src, drift, prm, pol)[1]
    for target in (0.5, 0.33):
        for sup in (False, True):
            tau = min(TAUS, key=lambda t: abs(src_budget[(t, sup)] - target))
            pol = (lambda t, sp: lambda sd, f, a, r: sd > t or (sp and f > 0.3))(tau, sup)
            m, b, pc = run(tgt, drift, prm, pol)
            out.append(dict(group=int(g), policy=f"VAR{int(target * 100)}{'+SUP' if sup else ''}", mae=m, budget=b, tau=tau,
                            src_budget=src_budget[(tau, sup)], cells=pc))
    return out


def ci(r, a, b, seed=67):
    A = {(x.group, c): v for x in r[r.policy == a].itertuples() for c, v in x.cells.items()}
    B = {(x.group, c): v for x in r[r.policy == b].itertuples() for c, v in x.cells.items()}
    T = pd.DataFrame([dict(group=k[0], a=A[k], b=B[k]) for k in A if k in B])
    rng = np.random.default_rng(seed)
    diffs = [np.mean([(lambda z: (z.a - z.b).mean())(x.sample(len(x), replace=True, random_state=int(rng.integers(1e9)))) for _, x in T.groupby("group")])
             for _ in range(2000)]
    return [round(float(v), 4) for v in np.percentile(diffs, [2.5, 97.5])]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=False)
    with Pool(13) as pool:
        rows = [x for o in pool.map(job, sorted(cg.unique())) for x in o]
    r = pd.DataFrame(rows)
    r.drop(columns="cells").to_csv(OUT / "per_group.csv", index=False)
    S = r.groupby("policy").agg(mae=("mae", "mean"), budget=("budget", "mean")).round(4).sort_values("budget")
    res = dict(table=S.reset_index().to_dict("records"))
    res["VAR50_minus_PER2_CI95"] = ci(r, "VAR50", "PER2")
    res["VAR33_minus_PER3_CI95"] = ci(r, "VAR33", "PER3")
    res["C1_pass"] = res["VAR50_minus_PER2_CI95"][1] < 0 and res["VAR33_minus_PER3_CI95"][1] < 0
    res["C2_pass"] = bool(S.loc["VAR50", "mae"] <= 1.05 * S.loc["ALL", "mae"] and S.loc["VAR50", "budget"] <= 0.6)
    res["VAR50_over_ALL"] = round(float(S.loc["VAR50", "mae"] / S.loc["ALL", "mae"]), 3)
    json.dump(res, open(OUT / "summary.json", "w"), indent=2)
    print(json.dumps(res, indent=1))
