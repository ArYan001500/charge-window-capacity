"""CALCE partial-cycling cells: window, throughput and reference-capacity tables for the leakage demonstration."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
WIN_S = int(sys.argv[3]) if len(sys.argv) > 3 else 600
OUT.mkdir(parents=True, exist_ok=False)
N_WIN, SEED, MAX_DELAY_H = 3, 20261026, 48.0
H22 = ROOT / "tables/calce_numeric"

steps = pd.read_csv(H22 / "step_runs_offline_provenance.csv")
links = pd.read_csv(H22 / "partial_charge_reference_timing_links.csv")
refs = pd.read_csv(H22 / "reference_discharge_candidates.csv")
refQ = refs[refs.protocol_matching_diagnostic_candidate == True].set_index("run_uid").Qd_counter_increment_Ah

charges = steps[(steps.is_partial_operation == True) & steps.I_mean_A.between(0.6, 0.9)
                & (steps.duration_observed_s > WIN_S + 120) & (steps.V_max <= 4.21)]
charges = charges.merge(links[["run_uid", "previous_reference_uid", "next_reference_uid",
                               "next_reference_delay_after_charge_hours"]], on="run_uid")
charges = charges[charges.previous_reference_uid.isin(refQ.index) & charges.next_reference_uid.isin(refQ.index)
                  & (charges.next_reference_delay_after_charge_hours <= MAX_DELAY_H)]

rng = np.random.default_rng(SEED)
rows = []
for cell in ["PL17", "PL21"]:
    blocks = sio.loadmat(H22 / f"{cell}_numeric.mat", squeeze_me=True, struct_as_record=False)["blocks"]
    by_op = {int(b.operation_index): b.data for b in blocks if hasattr(b, "data") and np.ndim(b.data) == 2}
    for _, r in charges[charges.cell_id == cell].iterrows():
        d = by_op[int(r.operation_index)][int(r.raw_first):int(r.raw_last)]
        t, i, v = d[:, 0], d[:, 4], d[:, 5]
        cc = i > 0.7
        t, i, v = t[cc], i[cc], v[cc]
        if len(t) < 20 or t[-1] - t[0] < WIN_S + 60 or np.any(np.diff(t) <= 0):
            continue
        for k in range(N_WIN):
            t0 = rng.uniform(t[0], t[-1] - WIN_S)
            m = (t >= t0) & (t <= t0 + WIN_S)
            tw, iw, vw = t[m], i[m], v[m]
            q = np.concatenate([[0], np.cumsum(np.diff(tw) * 0.5 * (iw[1:] + iw[:-1]) / 3600)])
            rows.append(dict(cell=cell, run_uid=r.run_uid, k=k, Q=refQ[r.next_reference_uid],
                             Q_prior=refQ[r.previous_reference_uid],
                             delay_h=r.next_reference_delay_after_charge_hours,
                             f0=vw[0], f1=vw[-1], f2=vw[-1] - vw[0], f3=np.polyfit(q, vw, 1)[0],
                             f4=np.polyfit(q, vw, 2)[0], f5=vw.mean()))
d = pd.DataFrame(rows)
F = [f"f{i}" for i in range(6)]


def design(X):
    X = np.asarray(X, float)
    iu = np.triu_indices(X.shape[1])
    return np.hstack([X, (X[:, :, None] * X[:, None, :])[:, iu[0], iu[1]]])


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Zs = (X - mu) / sd
    w = np.linalg.solve(Zs.T @ Zs + alpha * np.eye(Zs.shape[1]), Zs.T @ (y - y.mean()))
    return lambda Xn: (Xn - mu) / sd @ w + y.mean()


def pick_alpha(X, y, groups):
    folds = np.array_split(np.argsort(groups, kind="stable"), 5)
    def cv(a):
        return np.mean([np.abs(ridge_fit(np.delete(X, f, 0), np.delete(y, f), a)(X[f]) - y[f]).mean() for f in folds])
    return min(np.logspace(-4, 3, 22), key=cv)


out = []
for tr_cell, te_cell in [("PL17", "PL21"), ("PL21", "PL17")]:
    tr, te = d[d.cell == tr_cell], d[d.cell == te_cell].copy()
    Xtr, Xte = design(tr[F]), design(te[F])
    grp = tr.run_uid.astype("category").cat.codes.values
    a_abs = pick_alpha(Xtr, tr.Q.values, grp)
    a_res = pick_alpha(Xtr, (tr.Q - tr.Q_prior).values, grp)
    te["Q_train_mean"] = tr.Q.mean()
    te["Q_ridge"] = ridge_fit(Xtr, tr.Q.values, a_abs)(Xte)
    te["Q_prior_plus_resid"] = te.Q_prior + ridge_fit(Xtr, (tr.Q - tr.Q_prior).values, a_res)(Xte)
    te["Q_prior_plus_mean_resid"] = te.Q_prior + (tr.Q - tr.Q_prior).mean()
    te["ridge_boot_std"] = np.array([ridge_fit(Xtr[ii], tr.Q.values[ii], a_abs)(Xte)
                                     for ii in [rng.integers(0, len(tr), len(tr)) for _ in range(30)]]).std(0)
    te["train_cell"] = tr_cell
    out.append(te)
t = pd.concat(out)
t.to_csv(OUT / "scored_DEV.csv", index=False)
pol = ["Q_train_mean", "Q_prior", "Q_prior_plus_mean_resid", "Q_ridge", "Q_prior_plus_resid"]
res = {"n_windows": len(t), "n_charges": int(t.run_uid.nunique()), "win_s": WIN_S,
       "per_cell_counts": t.groupby("cell").size().to_dict(),
       "MAE": {p: {"eqcell": float((t[p] - t.Q).abs().groupby(t.cell).mean().mean()),
                   "cells": (t[p] - t.Q).abs().groupby(t.cell).mean().round(4).to_dict(),
                   "bias": float((t[p] - t.Q).mean())} for p in pol}}
ae = (t.Q_ridge - t.Q).abs()
o = t.ridge_boot_std.rank(method="first")
res["ridge_gate_risk_coverage"] = {str(c): float(ae[o <= c * len(t)].mean()) for c in (0.25, 0.5, 1.0)}
res["status"] = "GENUINE_PARTIAL_DEV_2CELLS"
json.dump(res, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(res, indent=1))
