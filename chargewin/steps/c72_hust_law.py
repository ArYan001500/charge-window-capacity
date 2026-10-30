"""HUST: semi-synthetic check of the trigger law."""
import json
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
C14 = ROOT / "results/C14_HUST_LOGO_2026-10-23_v1"
QG = [1e-8, 1e-7, 1e-6, 1e-5]
GRID = [(a, b) for a in QG for b in QG]
SIG = {"0": 0.0, "0.2%": 0.0022, "0.4%": 0.0043, "1.25%": 0.0135}
PERIODIC = {"PER3": lambda sc, a: (a - 1) % 3 == 0, "PER2": lambda sc, a: (a - 1) % 2 == 0,
            "PER3of5": lambda sc, a: (a - 1) % 5 in (0, 2, 3), "PER2of3": lambda sc, a: (a - 1) % 3 in (0, 1),
            "ALL": lambda sc, a: True}
DATA = {}


def load(mode):
    d = pd.read_csv(C14 / f"scored_{mode}.csv.gz", usecols=["cell", "cyc", "group", "Q", "MW_quad"]).rename(columns={"MW_quad": "pred"})
    d = d.sort_values(["cell", "cyc"]).reset_index(drop=True)
    first = d.groupby("cell").cyc.transform("min")
    d["slot"] = ((d.cyc - first) % 50 == 0).values
    return d


def events(d, sigma):
    ev = {}
    for c, x in d.groupby("cell"):
        e = []
        for i, cy, q, p, sl in zip(x.index, x.cyc, x.Q, x.pred, x.slot):
            if sl:
                rng = np.random.default_rng(zlib.crc32(f"{c}|{cy}".encode()))
                e.append((float(cy), True, q + rng.normal(0, sigma) if sigma > 0 else q, i, p))
            else:
                e.append((float(cy), False, p, i, p))
        ev[c] = e
    return ev


def kf(e, d0, prm, policy, slot):
    qn, qb, r_t, r_w, p_b0 = prm
    s, out, prev, tL, Q0, own = None, {}, None, None, None, None
    rate, used, avail, inn = d0, 0, 0, []
    for day, is_test, z, row, p in e:
        if s is not None:
            dt = day - prev
            s = s + np.array([rate * dt, 0.0])
            P = P + np.diag([qn * dt, qb * dt])
        prev = day
        if is_test:
            avail += 1
            take = s is None or policy({"INNOV": abs(np.mean(inn)) if inn else 0.0}, avail)
            if take:
                used += 1
                inn = []
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
            z = p
        if s is not None:
            H = np.array([1.0, 1.0])
            inn.append(z - H @ s)
            Kg = P @ H / (H @ P @ H + r_w)
            s, P = s + Kg * (z - H @ s), P - np.outer(Kg, H @ P)
            if not slot[row]:
                out[row] = s[0]
    return out, used, avail


def run(d, ev, cells, d0, prm, policy):
    res, u, a, sc = {}, 0, 0, []
    for c in cells:
        o, uu, aa = kf(ev[c], d0, prm, policy, d.slot.values)
        x = d.loc[list(o)]
        res[c] = (float((pd.Series(o) - x.Q).abs().mean()), uu / aa)
        u, a = u + uu, a + aa
    return res, u / a


def eq(res, cells):
    return float(np.mean([res[c][0] for c in cells]))


def job(args):
    mode, sk, g = args
    if mode not in DATA:
        DATA[mode] = load(mode)
    d = DATA[mode]
    sigma = SIG[sk]
    ev = events(d, sigma)
    cg = d.groupby("cell").group.first()
    src, tgt = list(cg[cg != g].index), list(cg[cg == g].index)
    s = d[d.cell.isin(src)]
    d0 = float(np.mean([np.polyfit(x.cyc, x.Q, 1)[0] for _, x in s.groupby("cell")]))
    bias = (s.pred - s.Q).groupby(s.cell).mean()
    p_b0 = float(np.var(bias) + np.mean(bias) ** 2)
    r_w = float(np.var((s.pred - s.Q) - s.cell.map(bias)))
    r_t = 1e-6 + sigma ** 2
    best = min(GRID, key=lambda q: eq(run(d, ev, src, d0, (q[0], q[1], r_t, r_w, p_b0), PERIODIC["ALL"])[0], src))
    prm = (best[0], best[1], r_t, r_w, p_b0)
    rows = []
    for name, pol in PERIODIC.items():
        res, _ = run(d, ev, tgt, d0, prm, pol)
        rows += [dict(mode=mode, sigma=sk, group=int(g), cell=c, policy=name, mae=m, budget=b) for c, (m, b) in res.items()]
    scs = []
    run(d, ev, src, d0, prm, lambda sc, a: scs.append(sc["INNOV"]) or True)
    taus = np.unique(np.quantile(scs, np.arange(0.05, 0.96, 0.05)))
    hyb = lambda t: (lambda sc, a: (a - 1) % 3 == 0 or sc["INNOV"] > t)
    sb = {t: run(d, ev, src, d0, prm, hyb(t))[1] for t in taus}
    tau = min(sb, key=lambda t: abs(sb[t] - 0.5))
    res, _ = run(d, ev, tgt, d0, prm, hyb(tau))
    rows += [dict(mode=mode, sigma=sk, group=int(g), cell=c, policy="TRIGGER", mae=m, budget=b, tau=tau) for c, (m, b) in res.items()]
    return rows


def cell_adv(x):
    per = x[x.policy != "TRIGGER"].sort_values("budget")
    tr = x[x.policy == "TRIGGER"].iloc[0]
    pb = per.groupby("budget").mae.mean()
    return float(np.interp(tr.budget, pb.index.values, pb.values) - tr.mae)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=False)
    groups = sorted(load("random").group.unique())
    jobs = [(m, s, g) for m in ("random", "fixed") for s in SIG for g in groups]
    with Pool(16) as pool:
        rows = [x for o in pool.map(job, jobs) for x in o]
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "per_cell.csv", index=False)
    A = r.groupby(["mode", "sigma", "group", "cell"]).apply(cell_adv).rename("A").reset_index()
    rng = np.random.default_rng(72)
    out = {}
    for (m, s), x in A.groupby(["mode", "sigma"]):
        bs = [np.mean([z.A.sample(len(z), replace=True, random_state=int(rng.integers(1e9))).mean() for _, z in x.groupby("group")]) for _ in range(2000)]
        tr = r[(r["mode"] == m) & (r.sigma == s)]
        out[f"{m}_{s}"] = dict(A=round(float(x.groupby("group").A.mean().mean()), 5), CI95=[round(float(v), 5) for v in np.percentile(bs, [2.5, 97.5])],
                               cells_A_pos=f"{int((x.A > 0).sum())}/{len(x)}",
                               trigger_budget=round(float(tr[tr.policy == "TRIGGER"].budget.mean()), 3),
                               trigger_mae=round(float(tr[tr.policy == "TRIGGER"].groupby("group").mae.mean().mean()), 5),
                               ALL_mae=round(float(tr[tr.policy == "ALL"].groupby("group").mae.mean().mean()), 5),
                               PER2_mae=round(float(tr[tr.policy == "PER2"].groupby("group").mae.mean().mean()), 5))
    g = lambda s: out[f"random_{s}"]
    res = dict(cells=out)
    res["H1_pass"] = g("0")["CI95"][0] > 0
    res["H2_pass"] = g("0.4%")["A"] < g("0")["A"] / 2 or g("0.4%")["CI95"][0] <= 0
    res["H3_pass"] = g("1.25%")["A"] <= 0 or g("1.25%")["CI95"][0] <= 0
    res["LAW_CONFIRMED"] = res["H1_pass"] and res["H2_pass"] and res["H3_pass"]
    json.dump(res, open(OUT / "summary.json", "w"), indent=2)
    print(json.dumps(res, indent=1))
