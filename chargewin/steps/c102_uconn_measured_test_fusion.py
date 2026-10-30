"""UConn fusion with the measured reference capacity as the test."""
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
QS = [(qn, 0.0) for qn in (1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3)]


def job(args):
    T, k, test = args
    src = [c for kk, cs in m.cells_by_group.items() if kk != k for c in cs]
    sw = m.nest["W"][k]
    st = m.nest["WR"][k] if test == "pulse" else m.y
    pr = m.fit_params(src, sw, st)
    if test == "measured":
        pr["rt"] = 1e-4
    src_groups = [kk for kk in m.groups if kk != k]
    best = {}
    for mode, grid in (("bias", QB), ("scalar", QS)):
        rw = pr["rw_in"] if mode == "bias" else pr["rw_all"]
        sc = []
        for qn, qb in grid:
            e = {c: m.run(c, sw, st, T, mode, pr["drift"], qn, qb, pr["rt"], rw, pr["pb0"]) for c in src}
            sc.append((np.mean([m.group_mae(m.cells_by_group[kk], e) for kk in src_groups]), qn, qb))
        best[mode] = min(sc, key=lambda s: s[0])
    tw = m.d.L_W.values
    tt = m.d.L_WR.values if test == "pulse" else m.y
    tc = m.cells_by_group[k]
    est = {"ref_drift": {c: m.run(c, tw, tt, T, "ref", pr["drift"], 0, 0, pr["rt"], 1, pr["pb0"]) for c in tc},
           "kf_scalar": {c: m.run(c, tw, tt, T, "scalar", pr["drift"], best["scalar"][1], 0, pr["rt"], pr["rw_all"], pr["pb0"]) for c in tc},
           "kf_bias": {c: m.run(c, tw, tt, T, "bias", pr["drift"], best["bias"][1], best["bias"][2], pr["rt"], pr["rw_in"], pr["pb0"]) for c in tc},
           "window": {c: m.run(c, tw, tt, T, "window", 0, 0, 0, 1, 1, 1) for c in tc}}
    r = dict(test=test, T=T, group=int(k), **{mm: m.group_mae(tc, e) for mm, e in est.items()})
    cells = [dict(test=test, T=T, group=int(k), cell=int(c), **{mm: m.cell_mae(est[mm][c]) for mm in est}) for c in tc]
    print(test, T, k, flush=True)
    return r, cells


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=False)
    m.prepare()
    st = {key: getattr(m, key) for key in ("d", "y", "g", "groups", "nest", "dev", "day", "tday", "cell_rows", "cells_by_group", "slope")}
    with open(OUT / "state.pkl", "wb") as f:
        pickle.dump(st, f)
    with Pool(16, initializer=m.load_state, initargs=(OUT / "state.pkl",)) as p:
        res = p.map(job, [(T, k, t) for t in ("pulse", "measured") for T in m.TS for k in m.groups], chunksize=1)
    (OUT / "state.pkl").unlink()
    pg = pd.DataFrame([r for r, _ in res])
    pc = pd.DataFrame([c for _, cs in res for c in cs])
    pg.to_csv(OUT / "per_group.csv", index=False)
    pc.to_csv(OUT / "per_cell.csv", index=False)
    c91 = pd.read_csv(m.R / "C91_UCONN_CAUSAL_FUSION_2026-10-29_v1/per_group.csv")
    c91 = c91[c91.variant == "causal_nested"].set_index(["T", "group"])
    q = pg[pg.test == "pulse"].set_index(["T", "group"])
    summ = {"S102_max_dev_vs_C91": float(max((q[c] - c91.loc[q.index, c]).abs().max() for c in ("ref_drift", "kf_scalar", "kf_bias", "window")))}
    summ["S102_pass"] = summ["S102_max_dev_vs_C91"] <= 1e-9
    for (test, T), x in pg.groupby(["test", "T"]):
        c = pc[(pc.test == test) & (pc["T"] == T)]
        summ[f"{test}_T{T}"] = dict(**{mm: round(float(x[mm].mean()), 4) for mm in ("ref_drift", "kf_scalar", "kf_bias", "window")},
                                    bias_beats_ref_groups=int((x.kf_bias < x.ref_drift).sum()), n_groups=len(x),
                                    ci_bias_minus_ref=m.boot_ci(c, "kf_bias", "ref_drift"),
                                    reduction_pct=round(100 * (1 - x.kf_bias.mean() / x.ref_drift.mean()), 1))
    json.dump(summ, open(OUT / "summary.json", "w"), indent=2)
    print(json.dumps(summ, indent=1))
