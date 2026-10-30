"""UConn: bias-aware Kalman fusion with pulse-based tests (first implementation)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

C19, C11, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=False)
a = pd.read_csv(C19 / "scored.csv.gz", parse_dates=["t"])
b = pd.read_csv(C11 / "scored_logo.csv.gz", parse_dates=["t"])[["cell", "t", "boot_std"]]
d = a.merge(b, on=["cell", "t"]).sort_values(["cell", "t"]).reset_index(drop=True)
d["day"] = (d.t - d.t.min()).dt.total_seconds() / 86400
slope = {c: np.polyfit(x.day, x.Q, 1)[0] for c, x in d.groupby("cell")}
cells_by_group = {g: list(x.cell.unique()) for g, x in d.groupby("group")}
arr = {c: (x.day.values, x.W.values, x["W+Rall"].values, x.Q.values) for c, x in d.groupby("cell")}


def run_cell(c, T, drift, qn, qb, r_test, r_win, p_b0):
    day, W, Wt, Q = arr[c]
    x = None
    est = np.empty(len(day))
    last_test, prev = None, day[0]
    for i in range(len(day)):
        if x is not None:
            dt = day[i] - prev
            x = x + np.array([drift * dt, 0.0])
            P = P + np.diag([qn * dt, qb * dt])
        prev = day[i]
        if last_test is None or day[i] - last_test >= T:
            if x is None:
                x, P = np.array([Wt[i], 0.0]), np.diag([r_test, p_b0])
            else:
                H = np.array([1.0, 0.0])
                S = H @ P @ H + r_test
                K = P @ H / S
                x, P = x + K * (Wt[i] - H @ x), P - np.outer(K, H @ P)
            last_test = day[i]
        else:
            H = np.array([1.0, 1.0])
            S = H @ P @ H + r_win
            K = P @ H / S
            x, P = x + K * (W[i] - H @ x), P - np.outer(K, H @ P)
        est[i] = x[0]
    return est


def score(cells, ests):
    return np.mean([np.mean(np.abs(ests[c] - arr[c][3])) for c in cells])


res, rows = {}, []
for T in (7, 14, 28, 56):
    maes = []
    for g in sorted(cells_by_group):
        src = [c for gg, cs in cells_by_group.items() if gg != g for c in cs]
        drift = float(np.mean([slope[c] for c in src]))
        s = d[d.cell.isin(src)]
        r_test = float(np.var(s["W+Rall"] - s.Q))
        cell_bias = (s.W - s.Q).groupby(s.cell).mean()
        p_b0 = float(np.var(cell_bias) + np.mean(cell_bias) ** 2)
        r_win = float(np.var((s.W - s.Q) - s.cell.map(cell_bias)))
        best, bp = np.inf, None
        for qn in (1e-5, 1e-4, 1e-3):
            for qb in (1e-6, 1e-5, 1e-4, 1e-3):
                m = score(src, {c: run_cell(c, T, drift, qn, qb, r_test, r_win, p_b0) for c in src})
                if m < best:
                    best, bp = m, (qn, qb)
        tc = cells_by_group[g]
        maes.append(score(tc, {c: run_cell(c, T, drift, bp[0], bp[1], r_test, r_win, p_b0) for c in tc}))
        rows.append(dict(T=T, group=g, qn=bp[0], qb=bp[1], mae=maes[-1]))
    res[f"T{T}_bias_kf"] = float(np.mean(maes))
    print(T, round(res[f"T{T}_bias_kf"], 4), flush=True)
pd.DataFrame(rows).to_csv(OUT / "per_group.csv", index=False)
json.dump(res, open(OUT / "summary.json", "w"), indent=2)
