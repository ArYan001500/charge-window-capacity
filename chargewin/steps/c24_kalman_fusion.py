"""UConn: scalar Kalman fusion of window estimates with pulse-based tests (first implementation)."""
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
arr = {c: (x.day.values, x.W.values, x["W+Rall"].values, x.boot_std.values, x.Q.values) for c, x in d.groupby("cell")}


def run_cell(c, T, drift, qn, r_test, rw_fn, use_windows):
    day, W, Wt, bs, Q = arr[c]
    x = P = None
    last_test, est = None, np.empty(len(day))
    prev_day = day[0]
    for i in range(len(day)):
        if x is not None:
            dt = day[i] - prev_day
            x, P = x + drift * dt, P + qn * dt
        prev_day = day[i]
        if last_test is None or day[i] - last_test >= T:
            if x is None:
                x, P = Wt[i], r_test
            else:
                k = P / (P + r_test)
                x, P = x + k * (Wt[i] - x), (1 - k) * P
            last_test = day[i]
        elif use_windows:
            rw = rw_fn(bs[i])
            k = P / (P + rw)
            x, P = x + k * (W[i] - x), (1 - k) * P
        est[i] = x
    return est


def score(cells, est_by_cell):
    return np.mean([np.mean(np.abs(est_by_cell[c] - arr[c][4])) for c in cells])


res, rows = {}, []
for T in (7, 14, 28, 56):
    for variant in ("ref_drift_only", "kf_const", "kf_boot"):
        maes = []
        for g in sorted(cells_by_group):
            src_cells = [c for gg, cs in cells_by_group.items() if gg != g for c in cs]
            drift = float(np.mean([slope[c] for c in src_cells]))
            s = d[d.cell.isin(src_cells)]
            r_test = float(np.var(s["W+Rall"] - s.Q))
            ew = (s.W - s.Q).values
            if variant == "kf_boot":
                A = np.c_[np.ones(len(s)), s.boot_std.values ** 2]
                coef = np.linalg.lstsq(A, ew ** 2, rcond=None)[0]
                rw_fn = lambda bstd, c0=max(coef[0], 1e-5), c1=max(coef[1], 0.0): c0 + c1 * bstd ** 2
            else:
                v = float(np.var(ew))
                rw_fn = lambda bstd, v=v: v
            use_w = variant != "ref_drift_only"
            best_q, best = 1e-6, np.inf
            for qn in (1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3):
                m = score(src_cells, {c: run_cell(c, T, drift, qn, r_test, rw_fn, use_w) for c in src_cells})
                if m < best:
                    best, best_q = m, qn
            tc = cells_by_group[g]
            maes.append(score(tc, {c: run_cell(c, T, drift, best_q, r_test, rw_fn, use_w) for c in tc}))
            rows.append(dict(T=T, variant=variant, group=g, qn=best_q, mae=maes[-1]))
        res[f"T{T}_{variant}"] = float(np.mean(maes))
        print(T, variant, round(res[f"T{T}_{variant}"], 4), flush=True)
pd.DataFrame(rows).to_csv(OUT / "per_group.csv", index=False)
json.dump(res, open(OUT / "summary.json", "w"), indent=2)
