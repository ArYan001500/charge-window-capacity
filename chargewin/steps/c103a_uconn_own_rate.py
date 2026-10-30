"""UConn fusion against propagation at each cell's own fade rate (slope between its last two released tests)."""
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c100_uconn_correlated_noise_filter as m

OUT = Path(sys.argv[2])
QB = [(qn, qb) for qn in (1e-5, 1e-4, 1e-3) for qb in (1e-6, 1e-5, 1e-4, 1e-3)]


def run_own(c, wpred, tpred, T, mode, d0, qn, qb, rt, rw, pb0):
    ev, _ = m.events(c, wpred, tpred, T)
    out = {}
    Q = b = p11 = p12 = p22 = None
    prev = None
    rate = d0
    last = None
    for t, kind, v in ev:
        if Q is not None:
            dt = t - prev
            Q += rate * dt
            p11 += qn * dt
            p22 += qb * dt
        prev = t
        if kind == 0:
            if mode == "ref" or Q is None:
                Q, b, p11, p12, p22 = v, 0.0, rt, 0.0, pb0
            else:
                S = p11 + rt
                k1, k2 = p11 / S, p12 / S
                inn = v - Q
                Q, b = Q + k1 * inn, b + k2 * inn
                p11, p12, p22 = p11 - k1 * p11, p12 - k1 * p12, p22 - k2 * p12
            if last is not None and t - last[0] > 0.5:
                rate = (v - last[1]) / (t - last[0])
            last = (t, v)
        else:
            if Q is None:
                continue
            if mode == "bias":
                z = wpred[v]
                h1, h2 = p11 + p12, p12 + p22
                S = h1 + h2 + rw
                k1, k2 = h1 / S, h2 / S
                inn = z - (Q + b)
                Q, b = Q + k1 * inn, b + k2 * inn
                p11, p12, p22 = p11 - k1 * h1, p12 - k1 * h2, p22 - k2 * h2
            out[v] = Q
    return out


def job(args):
    T, k = args
    src = [c for kk, cs in m.cells_by_group.items() if kk != k for c in cs]
    src_groups = [kk for kk in m.groups if kk != k]
    sw, st = m.nest["W"][k], m.nest["WR"][k]
    pr = m.fit_params(src, sw, st)

    def tune(runner):
        sc = []
        for qn, qb in QB:
            e = {c: runner(c, sw, st, T, "bias", pr["drift"], qn, qb, pr["rt"], pr["rw_in"], pr["pb0"]) for c in src}
            sc.append((np.mean([m.group_mae(m.cells_by_group[kk], e) for kk in src_groups]), qn, qb))
        return min(sc, key=lambda s: s[0])

    bp, bo = tune(m.run), tune(run_own)
    tw, tt = m.d.L_W.values, m.d.L_WR.values
    tc = m.cells_by_group[k]
    est = {"ref_drift": {c: m.run(c, tw, tt, T, "ref", pr["drift"], 0, 0, pr["rt"], 1, pr["pb0"]) for c in tc},
           "kf_bias": {c: m.run(c, tw, tt, T, "bias", pr["drift"], bp[1], bp[2], pr["rt"], pr["rw_in"], pr["pb0"]) for c in tc},
           "own_ref": {c: run_own(c, tw, tt, T, "ref", pr["drift"], 0, 0, pr["rt"], 1, pr["pb0"]) for c in tc},
           "own_kf": {c: run_own(c, tw, tt, T, "bias", pr["drift"], bo[1], bo[2], pr["rt"], pr["rw_in"], pr["pb0"]) for c in tc}}
    r = dict(T=T, group=int(k), **{mm: m.group_mae(tc, e) for mm, e in est.items()})
    cells = [dict(T=T, group=int(k), cell=int(c), **{mm: m.cell_mae(est[mm][c]) for mm in est}) for c in tc]
    print(T, k, flush=True)
    return r, cells


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=False)
    m.prepare()
    st = {key: getattr(m, key) for key in ("d", "y", "g", "groups", "nest", "dev", "day", "tday", "cell_rows", "cells_by_group", "slope")}
    with open(OUT / "state.pkl", "wb") as f:
        pickle.dump(st, f)
    with Pool(8, initializer=m.load_state, initargs=(OUT / "state.pkl",)) as p:
        res = p.map(job, [(T, k) for T in m.TS for k in m.groups], chunksize=1)
    (OUT / "state.pkl").unlink()
    pg = pd.DataFrame([r for r, _ in res])
    pc = pd.DataFrame([c for _, cs in res for c in cs])
    pg.to_csv(OUT / "per_group.csv", index=False)
    pc.to_csv(OUT / "per_cell.csv", index=False)
    c91 = pd.read_csv(m.R / "C91_UCONN_CAUSAL_FUSION_2026-10-29_v1/per_group.csv")
    c91 = c91[c91.variant == "causal_nested"].set_index(["T", "group"])
    q = pg.set_index(["T", "group"])
    summ = {"sanity_max_dev_vs_C91": float(max((q[c] - c91.loc[q.index, c]).abs().max() for c in ("ref_drift", "kf_bias")))}
    summ["sanity_pass"] = summ["sanity_max_dev_vs_C91"] <= 1e-9
    for T, x in pg.groupby("T"):
        c = pc[pc["T"] == T]
        summ[f"T{T}"] = dict(**{mm: round(float(x[mm].mean()), 4) for mm in ("ref_drift", "kf_bias", "own_ref", "own_kf")},
                             own_kf_beats_own_ref_groups=int((x.own_kf < x.own_ref).sum()),
                             kf_bias_beats_own_ref_groups=int((x.kf_bias < x.own_ref).sum()), n_groups=len(x),
                             ci_own_kf_minus_own_ref=m.boot_ci(c, "own_kf", "own_ref"),
                             ci_kf_bias_minus_own_ref=m.boot_ci(c, "kf_bias", "own_ref"))
    json.dump(summ, open(OUT / "summary.json", "w"), indent=2)
    print(json.dumps(summ, indent=1))
