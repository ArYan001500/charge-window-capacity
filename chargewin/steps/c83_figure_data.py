"""Data for the leakage figure."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]); OUT = ROOT / "results/C83_FIGURE_DATA_2026-10-29"
H22 = ROOT / "tables/calce_numeric"
s = pd.read_csv(ROOT / "results/C06_CALCE_GENUINE_PARTIAL_2026-10-22_w1800/scored_DEV.csv")
steps = pd.read_csv(H22 / "step_runs_offline_provenance.csv")[["run_uid", "Qc_counter_increment_Ah", "duration_observed_s"]]
d = s.merge(steps, on="run_uid", how="left").rename(columns={"Qc_counter_increment_Ah": "dq_charge", "duration_observed_s": "dur"})
F = [f"f{i}" for i in range(6)]
def ridge_fit(X, y, alpha=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-12; Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return lambda Xn: (Xn - mu) / sd @ w + y.mean()
rows = []
for name, cols in (("window_only", F), ("window_dq_dur", F + ["dq_charge", "dur"])):
    for tr_c, te_c in (("PL17", "PL21"), ("PL21", "PL17")):
        tr, te = d[d.cell == tr_c].dropna(subset=cols), d[d.cell == te_c].dropna(subset=cols)
        p = ridge_fit(tr[cols].values, tr.Q.values)(te[cols].values)
        rows += [dict(model=name, test_cell=te_c, Q=q, pred=v, dq=dq) for q, v, dq in zip(te.Q, p, te.dq_charge)]
pd.DataFrame(rows).to_csv(OUT / "calce_leak_predictions.csv", index=False)
c = pd.read_csv(ROOT / "tables/C38_UCONN_REF_CURVES_2026-10-24_v1/uconn_ref_chg_QofV.csv.gz")
QC = [x for x in c.columns if x.startswith("q")]; V = np.array([float(x[1:]) for x in QC])
rpt = pd.read_csv(ROOT / "tables/C10_UCONN_2026-10-23_v1/uconn_rpt_table.csv.gz")[["cell", "wk", "Q_ref"]]
c = c.merge(rpt, on=["cell", "wk"]).dropna(subset=["Q_ref"]).reset_index(drop=True)
out = []
for _, r in c.iterrows():
    qv = r[QC].values.astype(float); ok = ~np.isnan(qv)
    qq = np.maximum.accumulate(qv[ok]); qq = qq - qq[0]; vv = V[ok]
    def energy(a, b):
        if b > qq[-1]: return np.nan
        g = np.linspace(a, b, 200); return float(np.trapezoid(np.interp(g, qq, vv), g))
    out.append(dict(cell=r.cell, wk=r.wk, Q=r.Q_ref, E_soc=energy(0.3 * r.Q_ref, 0.7 * r.Q_ref), E_fixed=energy(0.675, 1.575)))
pd.DataFrame(out).dropna().to_csv(OUT / "soc_energy.csv", index=False)
print("ok")
