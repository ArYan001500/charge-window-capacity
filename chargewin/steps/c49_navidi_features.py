"""UConn: pulse features of the published pulse-based estimator."""
import json
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import kurtosis, skew

RPTZIP, ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])


def fit_2rc(t, v, i, r0):
    """V = V_ocv - i*R0 - V1 - V2 (sign: charging current positive raises V -> use +i). Discrete 1 s simulation."""
    v_oc = v[0] - i[0] * r0
    def sim(p):
        r1, c1, r2, c2 = p
        v1 = v2 = 0.0
        out = np.empty(len(t))
        for k in range(len(t)):
            dt = 1.0 if k == 0 else t[k] - t[k - 1]
            a1, a2 = np.exp(-dt / (r1 * c1)), np.exp(-dt / (r2 * c2))
            v1 = a1 * v1 + r1 * (1 - a1) * i[k]
            v2 = a2 * v2 + r2 * (1 - a2) * i[k]
            out[k] = v_oc + i[k] * r0 + v1 + v2
        return out
    p0 = [0.01, 100.0, 0.02, 2000.0]
    r = least_squares(lambda p: sim(p) - v, p0, bounds=([1e-3, 1, 1e-3, 1], [1, 1e4, 1, 1e4]), max_nfev=200)
    return r.x, float(np.abs(sim(r.x) - v).sum())


def one(name):
    z = zipfile.ZipFile(RPTZIP)
    cell = int(name.split("cell_")[1][:2])
    d = pd.read_csv(z.open(name), usecols=["Week Number", "Time (s)", "Voltage (V)", "Current (A)", "Segment Key", "Pulse Type", "Pulse SOC"])
    sp = d[(d["Segment Key"] == "slowpulse") & d["Pulse Type"].isin(["chg", "dchg"])]
    rows = []
    for (wk, typ, soc), s in sp.groupby(["Week Number", "Pulse Type", "Pulse SOC"]):
        t = s["Time (s)"].values - s["Time (s)"].values[0]
        v, i = s["Voltage (V)"].values, s["Current (A)"].values
        if len(t) < 90:
            continue
        g = np.arange(0, 100)
        vg, ig = np.interp(g, t, v), np.interp(g, t, i)
        di = abs(ig[31] - ig[29])
        r0 = abs(vg[31] - vg[30]) / di if di > 0.3 else np.nan
        try:
            (r1, c1, r2, c2), l1 = fit_2rc(g.astype(float), vg, ig, r0 if np.isfinite(r0) else 0.03)
        except Exception:
            r1 = c1 = r2 = c2 = l1 = np.nan
        rows.append(dict(cell=cell, wk=float(wk), typ=typ, soc=float(soc), vmin=vg.min(), vmax=vg.max(), vmean=vg.mean(),
                         vvar=vg.var(), vskew=float(skew(vg)), vkurt=float(kurtosis(vg)), R0=r0, R1=r1, C1=c1, R2=r2, C2=c2, L1=l1))
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=False)
    names = sorted(n for n in zipfile.ZipFile(RPTZIP).namelist() if n.endswith(".csv"))
    with Pool(18) as p:
        res = p.map(one, names, chunksize=1)
    F = pd.DataFrame([r for rr in res for r in rr])
    F.to_csv(OUT / "navidi_pulse_features.csv.gz", index=False)
    print("pulses", F.shape, flush=True)
