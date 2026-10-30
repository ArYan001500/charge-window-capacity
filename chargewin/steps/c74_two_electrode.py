"""UConn: two-electrode model fitted to 30-min charge segments."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
I = 0.75
OCP = pd.read_csv(ROOT / "tables/C74_HALFCELL_OCP/halfcell_ocp.csv")
c = pd.read_csv(ROOT / "tables/C38_UCONN_REF_CURVES_2026-10-24_v1/uconn_ref_chg_QofV.csv.gz")
QC = [x for x in c.columns if x.startswith("q")]
VG = np.array([float(x[1:]) for x in QC])
rpt = pd.read_csv(ROOT / "tables/C10_UCONN_2026-10-23_v1/uconn_rpt_table.csv.gz")[["cell", "wk", "Q_ref"]]
grp = pd.read_csv(ROOT / "data/uconn/cycling_parameters.csv", encoding="utf-8-sig").set_index("Cell ID")["Group Number"]
c = c.merge(rpt, on=["cell", "wk"]).dropna(subset=["Q_ref"]).reset_index(drop=True)
c["group"] = c.cell.map(grp)
FIRST = c.wk == c.groupby("cell").wk.transform("min")


def ocp(setname):
    p = OCP[(OCP.set == setname) & (OCP.electrode == "Positive")].sort_values("sto")
    n = OCP[(OCP.set == setname) & (OCP.electrode == "Negative")].sort_values("sto")
    return (lambda x: np.interp(x, p.sto, p.U)), (lambda y: np.interp(y, n.sto, n.U))


def curve(row):
    qv = row[QC].values.astype(float)
    ok = ~np.isnan(qv)
    v, q = VG[ok], np.maximum.accumulate(qv[ok])
    keep = np.r_[True, np.diff(q) > 1e-6]
    return q[keep] - q[keep][0], v[keep]


def model_v(th, q, Up, Un):
    Cp, Cn, x0, y0, R = th
    return Up(np.clip(x0 - q / Cp, 0.001, 0.999)) - Un(np.clip(y0 + q / Cn, 0.001, 0.999)) + I * R


def fit(q, v, Up, Un, lo, hi, starts):
    best = None
    for s0 in starts:
        s0 = np.clip(s0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
        r = least_squares(lambda th: model_v(th, q, Up, Un) - v, s0, bounds=(lo, hi), method="trf", max_nfev=400)
        if best is None or r.cost < best.cost:
            best = r
    return best.x, float(np.sqrt(2 * best.cost / len(q)))


def capacity(th, Up, Un):
    Cp, Cn, x0, y0, R = th
    ocv = lambda s: Up(np.clip(x0 - s / Cp, 0.001, 0.999)) - Un(np.clip(y0 + s / Cn, 0.001, 0.999))
    s_grid = np.linspace(-6, 6, 4001)
    vv = ocv(s_grid)
    try:
        i_top = np.where(vv >= 4.2)[0][0]
        s_top = s_grid[i_top]
        down = ocv(s_grid) + 0 * s_grid - I * R
        idx = np.where((s_grid < s_top) & (down <= 2.5))[0]
        s_cut = s_grid[idx[-1]] if len(idx) else s_grid[0]
        return float(s_top - s_cut)
    except IndexError:
        return np.nan


def starts_for(lo, hi):
    rng = np.random.default_rng(74)
    return [np.array(lo) + (np.array(hi) - np.array(lo)) * rng.uniform(0.15, 0.85, 5) for _ in range(6)]


LO = [0.5, 0.5, 0.001, 0.001, 0.0]
HI = [5.0, 5.0, 0.999, 0.999, 0.3]


def job_full(args):
    i, setname = args
    Up, Un = ocp(setname)
    q, v = curve(c.loc[i])
    th, rmse = fit(q, v, Up, Un, LO, HI, starts_for(LO, HI))
    return i, setname, th, rmse, capacity(th, Up, Un)


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


def job_seg(args):
    i, setname, fresh, pos = args
    Up, Un = ocp(setname)
    q, v = curve(c.loc[i])
    s = segment(q, v, pos)
    if s is None or len(s[0]) < 8:
        return i, pos, np.nan, np.nan
    lo = [0.6 * fresh[0], 0.6 * fresh[1], 0.001, 0.001, 0.0]
    hi = [1.1 * fresh[0], 1.1 * fresh[1], 0.999, 0.999, 0.3]
    th, rmse = fit(s[0], s[1], Up, Un, lo, hi, starts_for(lo, hi))
    return i, pos, capacity(th, Up, Un), rmse


def eqmae(df, col):
    x = df.dropna(subset=[col])
    return float((x[col] - x.Q_ref).abs().groupby(x.cell).mean().mean())


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=False)
    sets = ("Chen2020", "Mohtat2020", "Ecker2015")
    fresh_idx = list(c.index[FIRST])
    with Pool(18) as pool:
        fr = pool.map(job_full, [(i, s) for i in fresh_idx for s in sets])
    F = pd.DataFrame([dict(i=i, set=s, rmse=r) for i, s, _, r, _ in fr])
    F["group"] = c.loc[F.i, "group"].values
    choice = {}
    for g in sorted(c.group.unique()):
        choice[int(g)] = F[F.group != g].groupby("set").rmse.mean().idxmin()
    print("OCP set choice per group", choice, flush=True)
    jobs = [(i, choice[int(c.at[i, "group"])]) for i in c.index]
    with Pool(18) as pool:
        full = pool.map(job_full, jobs)
    c["S1"] = np.nan
    c["S1_rmse"] = np.nan
    th_by = {}
    for i, s, th, rmse, cap in full:
        c.at[i, "S1"], c.at[i, "S1_rmse"] = cap, rmse
        th_by[i] = th
    fresh_th = {c.at[i, "cell"]: th_by[i] for i in fresh_idx}
    seg_jobs = [(i, choice[int(c.at[i, "group"])], fresh_th[c.at[i, "cell"]], pos) for i in c.index for pos in ("LOW", "MID", "HIGH")]
    with Pool(18) as pool:
        seg = pool.map(job_seg, seg_jobs)
    for i, pos, cap, rmse in seg:
        c.at[i, f"S2_{pos}"] = cap
    c.drop(columns=QC).to_csv(OUT / "estimates.csv", index=False)
    res = dict(ocp_choice=choice, S1_MAE=round(eqmae(c, "S1"), 4), S1_voltage_rmse_mV=round(float(c.S1_rmse.mean() * 1000), 1),
               S1_bias=round(float((c.S1 - c.Q_ref).mean()), 4),
               S2={p: round(eqmae(c, f"S2_{p}"), 4) for p in ("LOW", "MID", "HIGH")},
               S2_n={p: int(c[f"S2_{p}"].notna().sum()) for p in ("LOW", "MID", "HIGH")})
    res["S1_pass"] = res["S1_MAE"] <= 0.05
    res["S2_weak_pass"] = res["S2"]["LOW"] < 0.082
    res["S2_strong_pass"] = res["S2"]["LOW"] <= 0.026
    json.dump(res, open(OUT / "summary.json", "w"), indent=2)
    print(json.dumps(res, indent=1))
