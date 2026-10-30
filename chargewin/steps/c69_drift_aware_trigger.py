"""KIT: test triggers from forecast disagreement."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import c67_fewer_tests as G

C = G.C
OUT = Path(sys.argv[2])
FMDIR = C.K.ROOT / "results/C69_FM_FORECASTS_2026-10-24_v1"
FM = {}
for m in ("chronos2", "moirai2"):
    F = pd.read_csv(FMDIR / f"fm_{m}.csv.gz")
    FM[m] = {(c, round(k, 4)): (g.d0.iloc[0], g.med.values) for (c, k), g in F.groupby(["cell", "kday"])}
SCORES = ["VAR", "INNOV", "FLUCT", "FM_chronos2", "FM_moirai2"]


def kf(c, drift, qn, qb, r_w, p_b0, policy, rng):
    ev, d = C.ev, C.d
    s, out, prev = None, {}, None
    st = dict(prev=None, tL=None, tP=None, Q0=None, now=None)
    rate, used, avail = 0.0, 0, 0
    inn, wins = [], []
    scores_seen = []
    for day, is_test, z, row in ev[c]:
        if s is not None:
            dt = day - prev
            s = s + np.array([rate * dt, 0.0])
            P = P + np.diag([qn * dt, qb * dt])
        prev = day
        if is_test:
            avail += 1
            if s is None:
                take = True
            else:
                sc = {"VAR": np.sqrt(P[0, 0]), "INNOV": abs(np.mean(inn)) if inn else 0.0}
                if len(wins) >= 4:
                    tt, pp = np.array(wins).T
                    sc["FLUCT"] = float(np.std(pp - np.polyval(np.polyfit(tt, pp, 1), tt)))
                else:
                    sc["FLUCT"] = 0.0
                for m in ("chronos2", "moirai2"):
                    f = FM[m].get((c, round(st["tL"], 4)))
                    if f is None or not wins:
                        sc[f"FM_{m}"] = 0.0
                    else:
                        d0, med = f
                        tt, pp = np.array(wins).T
                        h = np.floor(tt - d0).astype(int) - int(np.floor(st["tL"] - d0))
                        sc[f"FM_{m}"] = float(abs(np.mean(pp - np.interp(h, np.arange(1, len(med) + 1), med))))
                scores_seen.append(sc)
                take = policy(sc, avail, rng)
            if take:
                used += 1
                inn, wins = [], []
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
            inn.append(z - H @ s)
            wins.append((day, z))
            Kg = P @ H / (H @ P @ H + r_w)
            s, P = s + Kg * (z - H @ s), P - np.outer(Kg, H @ P)
            out[row] = s[0]
        if s is not None:
            st["now"] = day
            rate = drift(c, st)
    return out, used, avail, scores_seen


def run(cells, drift, prm, policy, seed=0, collect=False):
    rng = np.random.default_rng(seed)
    o, u, a, ss = {}, 0, 0, []
    for c in cells:
        oo, uu, aa, sc = kf(c, drift, *prm, policy, rng)
        o.update(oo)
        u, a = u + uu, a + aa
        ss += sc
    m, x = C.eqmae(o)
    pc = (pd.Series({k: o[k] for k in x.index}) - x.Q).abs().groupby(x.cell).mean().to_dict()
    return (m, u / a, pc, ss) if collect else (m, u / a, pc)


def thr(name, t):
    return lambda sc, a, r: sc[name] > t


def hyb(name, t):
    return lambda sc, a, r: (a - 1) % 3 == 0 or sc[name] > t


def job(g):
    drift, _ = C.make_drift("D1", g)
    cg, d = C.cg, C.d
    src, tgt = list(cg[cg != g].index), list(cg[cg == g].index)
    s = d[d.cell.isin(src) & d.score_row]
    bias = (s.pred - s.Q).groupby(s.cell).mean()
    p_b0 = float(np.var(bias) + np.mean(bias) ** 2)
    r_w = float(np.var((s.pred - s.Q) - s.cell.map(bias)))
    allp = lambda sc, a, r: True
    best = min(C.K.GRID, key=lambda p: run(src, drift, (p[0], p[1], r_w, p_b0), allp)[0])
    prm = (best[0], best[1], r_w, p_b0)
    out = []
    for name, pol in (("ALL", allp), ("PER2", lambda sc, a, r: (a - 1) % 2 == 0), ("PER3", lambda sc, a, r: (a - 1) % 3 == 0)):
        m, b, pc = run(tgt, drift, prm, pol)
        out.append(dict(group=int(g), policy=name, mae=m, budget=b, cells=pc))
    _, _, _, ss = run(src, drift, prm, allp, collect=True)
    S = pd.DataFrame(ss)
    cands = {}
    for sname in SCORES:
        taus = np.unique(np.quantile(S[sname], np.arange(0.05, 0.96, 0.05)))
        for kind, mk in (("thr", thr), ("hyb", hyb)):
            res = {t: run(src, drift, prm, mk(sname, t))[:2] for t in taus}
            for target in ((0.5, 0.33) if kind == "thr" else (0.5,)):
                t = min(res, key=lambda t: abs(res[t][1] - target))
                cands[(sname, kind, target)] = (t, res[t])
    for (sname, kind, target), (t, (smae, sbud)) in cands.items():
        m, b, pc = run(tgt, drift, prm, (thr if kind == "thr" else hyb)(sname, t))
        out.append(dict(group=int(g), policy=f"{sname}_{kind}{int(target * 100)}", mae=m, budget=b, src_mae=smae, src_budget=sbud, tau=t, cells=pc))
    for target in (0.5, 0.33):
        keys = [k for k in cands if k[2] == target]
        kbest = min(keys, key=lambda k: cands[k][1][0])
        r = next(o for o in out if o["policy"] == f"{kbest[0]}_{kbest[1]}{int(target * 100)}")
        out.append({**r, "policy": f"SELECTED{int(target * 100)}", "selected": f"{kbest[0]}_{kbest[1]}"})
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=False)
    with Pool(13) as pool:
        rows = [x for o in pool.map(job, sorted(C.cg.unique())) for x in o]
    r = pd.DataFrame(rows)
    r.drop(columns="cells").to_csv(OUT / "per_group.csv", index=False)
    Sm = r.groupby("policy").agg(mae=("mae", "mean"), budget=("budget", "mean")).round(4).sort_values("budget")
    res = dict(table=Sm.reset_index().to_dict("records"))
    res["D1_SELECTED50_minus_PER2_CI95"] = G.ci(r, "SELECTED50", "PER2")
    res["D1_SELECTED33_minus_PER3_CI95"] = G.ci(r, "SELECTED33", "PER3")
    res["D1_pass"] = res["D1_SELECTED50_minus_PER2_CI95"][1] < 0
    fm = [p for p in Sm.index if p.startswith("FM_") and p.endswith("50")]
    simple = [p for p in Sm.index if (p.startswith("INNOV") or p.startswith("FLUCT")) and p.endswith("50")]
    bf, bs = min(fm, key=lambda p: Sm.loc[p, "mae"]), min(simple, key=lambda p: Sm.loc[p, "mae"])
    res["D2_best_FM"], res["D2_best_simple"] = bf, bs
    res["D2_FM_minus_simple_CI95"] = G.ci(r, bf, bs)
    cand = [p for p in Sm.index if Sm.loc[p, "budget"] <= 0.6 and p != "ALL"]
    bb = min(cand, key=lambda p: Sm.loc[p, "mae"])
    res["D3_best_le60"], res["D3_ratio_to_ALL"] = bb, round(float(Sm.loc[bb, "mae"] / Sm.loc["ALL", "mae"]), 3)
    res["D3_pass"] = res["D3_ratio_to_ALL"] <= 1.05
    res["selected_counts"] = r[r.policy.str.startswith("SELECTED")].groupby("policy").selected.agg(lambda s: s.value_counts().to_dict()).to_dict()
    json.dump(res, open(OUT / "summary.json", "w"), indent=2, default=str)
    print(json.dumps(res, indent=1, default=str))
