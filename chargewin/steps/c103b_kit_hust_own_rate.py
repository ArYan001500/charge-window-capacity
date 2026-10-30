"""KIT and HUST fusion against propagation at each cell's own fade rate (slope between its last two released tests)."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c93_kit_hust_nested_fusion as N

K = N.K
ROOT, OUT = N.ROOT, N.OUT


def run_cell_own(x, labs, k_every, d0, qn, qb, r_test, r_win, p_b0, use_w):
    tests = labs.iloc[::k_every]
    ev = sorted([(t, "test", q) for t, q in zip(tests.t_lab, tests.Q)] +
                [(t, "win", i) for t, i in zip(x.t, x.index)], key=lambda e: e[0])
    s = None
    out = {}
    prev = None
    rate, last = d0, None
    for t, kind, val in ev:
        if s is not None:
            dt = (t - prev) / 86400
            s = s + np.array([rate * dt, 0.0])
            P = P + np.diag([qn * dt, qb * dt])
        prev = t
        if kind == "test":
            if s is None:
                s, P = np.array([val, 0.0]), np.diag([r_test, p_b0])
            else:
                H = np.array([1.0, 0.0])
                Kg = P @ H / (H @ P @ H + r_test)
                s, P = s + Kg * (val - H @ s), P - np.outer(Kg, H @ P)
            if last is not None and (t - last[0]) / 86400 > 0.5:
                rate = (val - last[1]) / ((t - last[0]) / 86400)
            last = (t, val)
        else:
            if s is None:
                continue
            if use_w:
                H = np.array([1.0, 1.0])
                Kg = P @ H / (H @ P @ H + r_win)
                s, P = s + Kg * (x.at[val, "pred"] - H @ s), P - np.outer(Kg, H @ P)
            out[val] = s[0]
    return out


def evaluate_own(cells, d, labs_by_cell, k_every, params, use_w):
    est = {}
    for c in cells:
        x = d[d.cell == c]
        est.update(run_cell_own(x, labs_by_cell[c], k_every, *params, use_w))
    e = d.loc[list(est)].assign(est=pd.Series(est))
    return e[e.near]


def job(args):
    g, k_every, d, labs, cbg = args
    src = [c for gg, cs in cbg.items() if gg != g for c in cs]
    s = d[d.cell.isin(src) & d.near]
    slope = [np.polyfit(labs[c].t_lab / 86400, labs[c].Q, 1)[0] for c in src if len(labs[c]) >= 2]
    drift = float(np.mean(slope))
    bias = (s.pred - s.Q).groupby(s.cell).mean()
    p_b0 = float(np.var(bias) + np.mean(bias) ** 2)
    r_win = float(np.var((s.pred - s.Q) - s.cell.map(bias)))
    out = dict(group=int(g), k_every=k_every)
    for name, use_w, ev in (("ref_drift", False, K.evaluate), ("kf_bias", True, K.evaluate),
                            ("own_ref", False, evaluate_own), ("own_kf", True, evaluate_own)):
        best, bp = np.inf, None
        for qn in (1e-5, 1e-4, 1e-3):
            for qb in ((1e-5, 1e-4, 1e-3) if use_w else (1e-5,)):
                m = K.score(ev(src, d, labs, k_every, (drift, qn, qb, 1e-4, r_win, p_b0), use_w))
                if m < best:
                    best, bp = m, (qn, qb)
        e = ev(cbg[g], d, labs, k_every, (drift, bp[0], bp[1], 1e-4, r_win, p_b0), use_w)
        out[name] = float((e.est - e.Q).abs().groupby(e.cell).mean().mean())
    return out


def run_dataset(tag, d, labs, X, train_mask, locked, k_list, pool):
    y, g = d.Q.values, d.group.values
    S = N.Stats(X, y, g, train_mask)
    groups = np.unique(g)
    cbg = {k: list(np.unique(d.cell[g == k])) for k in groups}
    jobs = []
    for k in groups:
        p = locked.copy()
        for h in groups:
            if h != k:
                p[g == h] = S.predict([j for j in groups if j not in (k, h)], X[g == h])
        jobs += [(k, ke, d.assign(pred=p), labs, cbg) for ke in k_list]
    r = pd.DataFrame(pool.map(job, jobs))
    r["dataset"] = tag
    print(tag, "done", flush=True)
    return r


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=False)
    rows = []
    with Pool(12) as pool:
        d, labs, X, tm = N.load_kit()
        locked = pd.read_csv(ROOT / "results/C32_KIT_KALMAN_2026-10-24_v1/window_predictions.csv.gz")
        d = d.merge(locked[["cell", "t", "pred"]], on=["cell", "t"], how="left")
        rows.append(run_dataset("KIT", d, labs, X, tm, d.pred.values, (1, 2), pool))
        for mode in ("fixed", "random"):
            for ke in (1, 2):
                d, labs, X, tm = N.load_hust(mode, 50)
                rows.append(run_dataset(f"HUST_{mode}", d, labs, X, tm, d.MW_quad.values.copy(), (ke,), pool))
    r = pd.concat(rows)
    r.to_csv(OUT / "per_group.csv", index=False)
    ref = pd.read_csv(ROOT / "results/C93_KIT_HUST_NESTED_FUSION_2026-10-29_v1/per_group.csv")
    a = r.set_index(["dataset", "k_every", "group"]).sort_index()
    b = ref.rename(columns={"bias_kf": "kf_bias"}).set_index(["dataset", "k_every", "group"]).sort_index()
    summ = {"sanity_max_dev_vs_C93": float(max((a[c] - b.loc[a.index, c]).abs().max() for c in ("ref_drift", "kf_bias")))}
    summ["sanity_pass"] = summ["sanity_max_dev_vs_C93"] <= 1e-9
    for (ds, ke), x in r.groupby(["dataset", "k_every"]):
        summ[f"{ds}_k{ke}"] = dict(**{m: round(float(x[m].mean()), 4) for m in ("ref_drift", "kf_bias", "own_ref", "own_kf")},
                                   own_kf_beats_own_ref=f"{int((x.own_kf < x.own_ref).sum())}/{len(x)}",
                                   kf_bias_beats_own_ref=f"{int((x.kf_bias < x.own_ref).sum())}/{len(x)}")
    json.dump(summ, open(OUT / "summary.json", "w"), indent=2)
    print(json.dumps(summ, indent=1))
