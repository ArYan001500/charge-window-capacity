"""UConn: two-electrode model with constrained electrode parameters."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
I, VTOP, VCUT, QNOM = 0.75, 4.2, 2.5, 2.25
OCP = pd.read_csv(ROOT / "tables/C74_HALFCELL_OCP/halfcell_ocp.csv")
P = OCP[(OCP.set == "Mohtat2020") & (OCP.electrode == "Positive")].sort_values("sto")
N = OCP[(OCP.set == "Mohtat2020") & (OCP.electrode == "Negative")].sort_values("sto")
Up = lambda x: np.interp(np.clip(x, 0.001, 0.999), P.sto, P.U)
Un = lambda y: np.interp(np.clip(y, 0.001, 0.999), N.sto, N.U)
_yn, _un = N.sto.values, np.minimum.accumulate(N.U.values)
Un_inv = lambda u: float(np.interp(-u, -_un, _yn))

c = pd.read_csv(ROOT / "tables/C38_UCONN_REF_CURVES_2026-10-24_v1/uconn_ref_chg_QofV.csv.gz")
QC = [x for x in c.columns if x.startswith("q")]
VG = np.array([float(x[1:]) for x in QC])
rpt = pd.read_csv(ROOT / "tables/C10_UCONN_2026-10-23_v1/uconn_rpt_table.csv.gz")[["cell", "wk", "Q_ref"]]
grp = pd.read_csv(ROOT / "data/uconn/cycling_parameters.csv", encoding="utf-8-sig").set_index("Cell ID")["Group Number"]
c = c.merge(rpt, on=["cell", "wk"]).dropna(subset=["Q_ref"]).reset_index(drop=True)
c["group"] = c.cell.map(grp)
FIRST = c.wk == c.groupby("cell").wk.transform("min")
S = np.linspace(0, 4.0, 2001)


def ocv(th):
    Cp, Cn, xt = th[:3]
    yt = Un_inv(Up(xt) - VTOP)
    return Up(xt + S / Cp) - Un(yt - S / Cn)


def capacity(th):
    v = ocv(th) - I * th[3]
    idx = np.where(v <= VCUT)[0]
    return float(S[idx[0]]) if len(idx) else float(S[-1])


def curve(row):
    qv = row[QC].values.astype(float)
    ok = ~np.isnan(qv)
    v, q = VG[ok], np.maximum.accumulate(qv[ok])
    keep = np.r_[True, np.diff(q) > 1e-6]
    return q[keep] - q[keep][0], v[keep]


def predict_charge(th, q, v1):
    """Charging from the state whose model voltage equals v1 (position fixed by the measured start voltage)."""
    vs = ocv(th) + I * th[3]
    cap = capacity(th)
    m = S <= max(cap, 0.05)
    s1 = float(np.interp(-v1, -np.minimum.accumulate(vs[m]), S[m]))
    s = s1 - q
    return np.interp(s, S, vs, left=vs[0] + 0.5, right=vs[-1])


def fit(q, v, lo, hi, prior=None, lam=0.0, nominal=False, starts=6):
    rng = np.random.default_rng(75)
    lo, hi = np.array(lo, float), np.array(hi, float)

    def res(th):
        r = predict_charge(th, q, v[0]) - v
        extra = []
        if nominal:
            extra.append(5.0 * (capacity(th) - QNOM))
        if prior is not None and lam > 0:
            extra += list(np.sqrt(lam) * len(q) ** 0.5 * (th[:3] / prior[:3] - 1))
        return np.r_[r, extra]
    best = None
    for _ in range(starts):
        s0 = lo + (hi - lo) * rng.uniform(0.2, 0.8, len(lo))
        try:
            r = least_squares(res, s0, bounds=(lo, hi), max_nfev=300)
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    th = best.x
    rmse = float(np.sqrt(np.mean((predict_charge(th, q, v[0]) - v) ** 2)))
    return th, rmse


def segment(q, v, pos, dq=0.375):
    if q[-1] < dq + 0.01:
        return None
    if pos == "LOW":
        a = 0.0
    elif pos == "HIGH":
        a = q[-1] - dq
    else:
        if v[0] > 3.75 or v[-1] < 3.75:
            return None
        a = np.interp(3.75, v, q) - dq / 2
        if a < 0 or a + dq > q[-1]:
            return None
    m = (q >= a) & (q <= a + dq)
    return q[m] - a, v[m]


def job_fresh(i):
    q, v = curve(c.loc[i])
    th, rmse = fit(q, v, [1.0, 1.0, 0.01, 0.0], [6.0, 6.0, 0.9, 0.3], nominal=True, starts=10)
    return c.at[i, "cell"], th, rmse, capacity(th)


def job_seg(args):
    i, pos, fresh = args
    q, v = curve(c.loc[i])
    s = segment(q, v, pos)
    if s is None or len(s[0]) < 8:
        return i, pos, {}
    lo = [0.5 * fresh[0], 0.5 * fresh[1], max(0.001, fresh[2] - 0.3), 0.0]
    hi = [1.05 * fresh[0], 1.05 * fresh[1], min(0.95, fresh[2] + 0.3), 0.3]
    out = {}
    for lam in (0.0, 1e-4, 1e-3, 1e-2):
        th, _ = fit(s[0], s[1], lo, hi, prior=fresh, lam=lam, starts=4)
        out[lam] = capacity(th)
    return i, pos, out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=False)
    with Pool(18) as pool:
        fr = pool.map(job_fresh, list(c.index[FIRST]))
    fresh = {cell: th for cell, th, _, _ in fr}
    sanity = dict(fresh_voltage_rmse_mV=round(float(np.mean([r for _, _, r, _ in fr]) * 1000), 1),
                  fresh_capacity_mean=round(float(np.mean([k for _, _, _, k in fr])), 3))
    print(sanity, flush=True)
    jobs = [(i, pos, fresh[c.at[i, "cell"]]) for i in c.index for pos in ("LOW", "MID", "HIGH")]
    with Pool(18) as pool:
        seg = pool.map(job_seg, jobs)
    rows = [dict(i=i, pos=pos, lam=lam, cap=v) for i, pos, out in seg for lam, v in out.items()]
    R = pd.DataFrame(rows).merge(c[["cell", "group", "wk", "Q_ref"]], left_on="i", right_index=True)
    R.to_csv(OUT / "segment_estimates.csv", index=False)
    eq = lambda x: float((x.cap - x.Q_ref).abs().groupby(x.cell).mean().mean())
    res = dict(sanity=sanity, by_position={}, lambda_choice={})
    for pos, x in R.groupby("pos"):
        preds = []
        for g in sorted(c.group.unique()):
            src, tgt = x[x.group != g], x[x.group == g]
            lam = min(src.lam.unique(), key=lambda l: eq(src[src.lam == l]))
            res["lambda_choice"].setdefault(pos, {})[int(g)] = lam
            preds.append(tgt[tgt.lam == lam])
        P_ = pd.concat(preds)
        res["by_position"][pos] = dict(MAE=round(eq(P_), 4), bias=round(float((P_.cap - P_.Q_ref).mean()), 4),
                                       corr=round(float(P_.cap.corr(P_.Q_ref)), 3), n=len(P_),
                                       src_MAE=round(float(np.mean([eq(x[(x.group != g) & (x.lam == res['lambda_choice'][pos][int(g)])]) for g in sorted(c.group.unique())])), 4))
    best = min(res["by_position"], key=lambda p: res["by_position"][p]["src_MAE"])
    res["best_position_by_source"] = best
    res["K_a_pass"] = res["by_position"][best]["MAE"] < 0.082
    res["K_b_pass_any"] = any(v["MAE"] <= 0.026 for v in res["by_position"].values())
    json.dump(res, open(OUT / "summary.json", "w"), indent=2)
    print(json.dumps(res, indent=1))
