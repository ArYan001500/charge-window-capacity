"""CALCE: accuracy gained by adding charge throughput to the window inputs."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=False)
H22 = ROOT / "tables/calce_numeric"
s = pd.read_csv(ROOT / "results/C06_CALCE_GENUINE_PARTIAL_2026-10-22_w1800/scored_DEV.csv")
steps = pd.read_csv(H22 / "step_runs_offline_provenance.csv")[["run_uid", "Qc_counter_increment_Ah", "duration_observed_s"]]
d = s.merge(steps, on="run_uid", how="left").rename(columns={"Qc_counter_increment_Ah": "dq_charge", "duration_observed_s": "dur"})
F = [f"f{i}" for i in range(6)]


def ridge_fit(X, y, alpha=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Zs = (X - mu) / sd
    w = np.linalg.solve(Zs.T @ Zs + alpha * np.eye(Zs.shape[1]), Zs.T @ (y - y.mean()))
    return lambda Xn: (Xn - mu) / sd @ w + y.mean()


res = {}
for name, cols in (("window_only", F), ("window+dq+duration (leaky)", F + ["dq_charge", "dur"]), ("dq_only (leaky)", ["dq_charge"])):
    maes = {}
    for tr_c, te_c in (("PL17", "PL21"), ("PL21", "PL17")):
        tr, te = d[d.cell == tr_c].dropna(subset=cols), d[d.cell == te_c].dropna(subset=cols)
        p = ridge_fit(tr[cols].values, tr.Q.values)(te[cols].values)
        maes[te_c] = float(np.abs(p - te.Q.values).mean())
    res[name] = dict(MAE_by_test_cell={k: round(v, 4) for k, v in maes.items()}, mean=round(float(np.mean(list(maes.values()))), 4))
res["prior_last_reference_MAE"] = round(float(d.groupby("cell").apply(lambda z: (z.Q_prior - z.Q).abs().mean()).mean()), 4)
json.dump(res, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(res, indent=1))
