"""UConn: test-on-demand trigger."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
GRID = [(qn, qb) for qn in (1e-5, 1e-4, 1e-3) for qb in (1e-6, 1e-5, 1e-4, 1e-3)]
c19 = sorted((ROOT / "results").glob("C19_*"))[0]
d = pd.read_csv(c19 / "scored.csv.gz", parse_dates=["t"]).sort_values(["cell", "t"]).reset_index(drop=True)
d["day"] = (d.t - d.t.min()).dt.total_seconds() / 86400
d = d.rename(columns={"W": "pred", "W+Rall": "wt"})
slopes = {c: np.polyfit(x.day, x.Q, 1)[0] for c, x in d.groupby("cell")}
cg = d.groupby("cell").group.first()
EV, SLOT = {}, np.zeros(len(d), bool)
for c, x in d.groupby("cell"):
    e, nxt = [], x.day.iloc[0]
    for i, day, w, wt in zip(x.index, x.day, x.pred, x.wt):
        if day >= nxt:
            e.append((day, True, wt, i))
            SLOT[i] = True
            nxt = day + 7
        else:
            e.append((day, False, w, i))
    EV[c] = e
d["score_row"] = ~SLOT


def kf(c, d0, prm, policy, rng):
    qn, qb, r_t, r_w, p_b0 = prm
    s, out, prev = None, {}, None
    tL = Q0 = own = None
    rate, used, avail, inn, wins = d0, 0, 0, [], []
    for day, is_test, z, row in EV[c]:
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
                sc = {"INNOV": abs(np.mean(inn)) if inn else 0.0}
                if len(wins) >= 4:
                    tt, pp = np.array(wins).T
                    sc["FLUCT"] = float(np.std(pp - np.polyval(np.polyfit(tt, pp, 1), tt)))
                else:
                    sc["FLUCT"] = 0.0
                take = policy(sc, avail, rng)
            if take:
                used += 1
                inn, wins = [], []
                if s is None:
                    s, P = np.array([z, 0.0]), np.diag([r_t, p_b0])
                else:
                    H = np.array([1.0, 0.0])
                    Kg = P @ H / (H @ P @ H + r_t)
                    s, P = s + Kg * (z - H @ s), P - np.outer(Kg, H @ P)
                    if day - tL > 0.5:
                        own = (z - Q0) / (day - tL)
                tL, Q0 = day, z
                rate = own if own is not None else d0
                continue
        if s is not None:
            H = np.array([1.0, 1.0])
            inn.append(z - H @ s)
            wins.append((day, z))
            Kg = P @ H / (H @ P @ H + r_w)
            s, P = s + Kg * (z - H @ s), P - np.outer(Kg, H @ P)
            if not SLOT[row]:
                out[row] = s[0]
    return out, used, avail


def run(cells, d0, prm, policy, seed=0):
    rng = np.random.default_rng(seed)
    o, u, a = {}, 0, 0
    for c in cells:
        oo, uu, aa = kf(c, d0, prm, policy, rng)
        o.update(oo)
        u, a = u + uu, a + aa
    x = d.loc[list(o)]
    e = (pd.Series(o) - x.Q).abs()
    pc = e.groupby(x.cell).mean()
    return float(pc.mean()), u / a, pc.to_dict()


PERIODIC = {"ALL": lambda sc, a, r: True, "PER2": lambda sc, a, r: (a - 1) % 2 == 0, "PER3": lambda sc, a, r: (a - 1) % 3 == 0,
            "PER3of5": lambda sc, a, r: (a - 1) % 5 in (0, 2, 3), "PER2of3": lambda sc, a, r: (a - 1) % 3 in (0, 1)}


def job(g):
    src, tgt = list(cg[cg != g].index), list(cg[cg == g].index)
    s = d[d.cell.isin(src)]
    d0 = float(np.mean([slopes[c] for c in src]))
    r_t = float(np.var(s.wt - s.Q))
    bias = (s.pred - s.Q).groupby(s.cell).mean()
    p_b0 = float(np.var(bias) + np.mean(bias) ** 2)
    r_w = float(np.var((s.pred - s.Q) - s.cell.map(bias)))
    best = min(GRID, key=lambda p: run(src, d0, (p[0], p[1], r_t, r_w, p_b0), PERIODIC["ALL"])[0])
    prm = (best[0], best[1], r_t, r_w, p_b0)
    out = []
    for name, pol in PERIODIC.items():
        m, b, pc = run(tgt, d0, prm, pol)
        out.append(dict(group=int(g), policy=name, mae=m, budget=b, cells=pc))
    scs = []
    collect = lambda sc, a, r: scs.append(sc) or True
    run(src, d0, prm, collect)
    S = pd.DataFrame(scs)
    cands = {}
    for sname in ("INNOV", "FLUCT"):
        taus = np.unique(np.quantile(S[sname], np.arange(0.05, 0.96, 0.05)))
        for kind in ("thr", "hyb"):
            mk = (lambda n, t: lambda sc, a, r: sc[n] > t) if kind == "thr" else (lambda n, t: lambda sc, a, r: (a - 1) % 3 == 0 or sc[n] > t)
            res = {t: run(src, d0, prm, mk(sname, t))[:2] for t in taus}
            t = min(res, key=lambda t: abs(res[t][1] - 0.5))
            m, b, pc = run(tgt, d0, prm, mk(sname, t))
            cands[f"{sname}_{kind}"] = res[t][0]
            out.append(dict(group=int(g), policy=f"{sname}_{kind}", mae=m, budget=b, tau=t, src_mae=res[t][0], src_budget=res[t][1], cells=pc))
    kbest = min(cands, key=cands.get)
    r = next(o for o in out if o["policy"] == kbest)
    out.append({**r, "policy": "SELECTED", "selected": kbest})
    passive = d[d.cell.isin(tgt) & d.score_row]
    out.append(dict(group=int(g), policy="PASSIVE", mae=float((passive.pred - passive.Q).abs().groupby(passive.cell).mean().mean()), budget=0.0, cells={}))
    return out


def ci(r, a, b, seed=70):
    A = {(x.group, c): v for x in r[r.policy == a].itertuples() for c, v in x.cells.items()}
    B = {(x.group, c): v for x in r[r.policy == b].itertuples() for c, v in x.cells.items()}
    T = pd.DataFrame([dict(group=k[0], a=A[k], b=B[k]) for k in A if k in B])
    rng = np.random.default_rng(seed)
    diffs = [np.mean([(lambda z: (z.a - z.b).mean())(x.sample(len(x), replace=True, random_state=int(rng.integers(1e9)))) for _, x in T.groupby("group")])
             for _ in range(2000)]
    return [round(float(v), 4) for v in np.percentile(diffs, [2.5, 97.5])]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=False)
    with Pool(11) as pool:
        rows = [x for o in pool.map(job, sorted(cg.unique())) for x in o]
    r = pd.DataFrame(rows)
    r.drop(columns="cells").to_csv(OUT / "per_group.csv", index=False)
    Sm = r.groupby("policy").agg(mae=("mae", "mean"), budget=("budget", "mean")).round(4).sort_values("budget")
    res = dict(table=Sm.reset_index().to_dict("records"))
    bsel = Sm.loc["SELECTED", "budget"]
    ref = min([p for p in ("PER2", "PER3of5", "PER2of3", "ALL") if Sm.loc[p, "budget"] >= bsel], key=lambda p: Sm.loc[p, "budget"])
    res["U1_reference"] = ref
    res["U1_CI95"] = ci(r, "SELECTED", ref)
    res["U1_pass"] = res["U1_CI95"][1] < 0
    pg = r[r.policy == "SELECTED"].set_index("group").budget
    pp = r[r.policy == "PASSIVE"].set_index("group").mae
    res["U2_spearman_budget_vs_passive_error"] = round(float(pg.corr(pp, method="spearman")), 3)
    res["U2_pass"] = res["U2_spearman_budget_vs_passive_error"] > 0
    res["U3_ratio_to_ALL"] = round(float(Sm.loc["SELECTED", "mae"] / Sm.loc["ALL", "mae"]), 3)
    res["U3_pass"] = bool(res["U3_ratio_to_ALL"] <= 1.05 and bsel <= 0.6)
    res["selected_counts"] = r[r.policy == "SELECTED"].selected.value_counts().to_dict()
    res["per_group_selected"] = r[r.policy == "SELECTED"][["group", "mae", "budget"]].round(3).to_dict("records")
    json.dump(res, open(OUT / "summary.json", "w"), indent=2)
    print(json.dumps(res, indent=1))
