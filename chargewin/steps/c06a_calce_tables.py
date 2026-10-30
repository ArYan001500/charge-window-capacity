"""CALCE PL17 and PL21: step runs, reference-discharge candidates and their timing relative to each partial charge."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(sys.argv[1])
O = ROOT / "tables/calce_numeric"
steps = []
for cell in ["PL17", "PL21"]:
    blocks = loadmat(O / f"{cell}_numeric.mat", simplify_cells=True)["blocks"]
    for rec in blocks:
        op = int(rec["operation_index"])
        x = np.asarray(rec["data"], dtype=float)
        base = {"cell_id": cell, "operation_index": op, "operation": str(rec["operation"]),
                "declared_start_date": str(rec["start_date"])}
        if not x.size:
            continue
        assert x.ndim == 2 and x.shape[1] == 8
        boundaries = np.r_[0, np.where(np.any(np.diff(x[:, 2:4], axis=0) != 0, axis=1))[0] + 1, len(x)]
        for j, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            a, b = int(a), int(b)
            y = x[a:b]
            dt = np.diff(y[:, 0])
            mI = float(y[:, 4].mean())
            span = float(y[:, 5].max() - y[:, 5].min())
            lo = max(0, a - 1)
            xx = x[lo:b]
            delta = np.diff(xx[:, 0])
            finite = bool(np.isfinite(xx).all())
            monotonic = bool((delta >= 0).all())
            right = float(np.dot(xx[1:, 4], delta) / 3600) if finite and monotonic else None
            trap = float(np.dot((xx[1:, 4] + xx[:-1, 4]) / 2, delta) / 3600) if finite and monotonic else None
            qd = float(y[-1, 7] - x[lo, 7])
            qc = float(y[-1, 6] - x[lo, 6])
            candidate = mI < -.02 and span >= 1
            matched = candidate and abs(mI + .75) <= .03 and y[0, 5] >= 4.05 and y[-1, 5] <= 2.85
            steps.append(dict(base, run_index=j, raw_first=a, raw_last=b - 1, step=int(y[0, 2]), cycle=int(y[0, 3]),
                              samples=len(y), start_native_s=float(y[0, 0]), end_native_s=float(y[-1, 0]),
                              duration_observed_s=float(y[-1, 0] - y[0, 0]),
                              measured_start_datenum=float(y[0, 1]), measured_end_datenum=float(y[-1, 1]),
                              V_start=float(y[0, 5]), V_end=float(y[-1, 5]), V_min=float(y[:, 5].min()),
                              V_max=float(y[:, 5].max()), I_mean_A=mI, I_min_A=float(y[:, 4].min()),
                              I_max_A=float(y[:, 4].max()), Qd_counter_increment_Ah=qd, Qc_counter_increment_Ah=qc,
                              signed_integral_right_Ah=right, signed_integral_trapezoid_Ah=trap,
                              Qd_counter_vs_right_abs_error_Ah=abs(qd + right) if right is not None and mI < -.02 else None,
                              native_dt_zero=int((dt == 0).sum()), native_dt_negative=int((dt < 0).sum()),
                              max_native_dt_s=float(delta.max()) if len(delta) else None,
                              prior_row_included=lo < a, broad_reference_candidate=candidate,
                              protocol_matching_diagnostic_candidate=matched,
                              is_partial_operation="partial" in str(rec["operation"]).lower(),
                              actual_received_at="UNKNOWN", label_assigned=False))
st = pd.DataFrame(steps)
st.insert(0, "run_uid", [f"{r.cell_id}_op{r.operation_index}_run{r.run_index}" for r in st.itertuples()])
st.to_csv(O / "step_runs_offline_provenance.csv", index=False)
ref = st[st.protocol_matching_diagnostic_candidate].copy()
ref.to_csv(O / "reference_discharge_candidates.csv", index=False)
pc = st[(st.is_partial_operation) & (((st.cell_id == "PL17") & (st.step == 7)) | ((st.cell_id == "PL21") & (st.step == 6)))].copy()
links = []
for row in pc.itertuples():
    prior = ref[(ref.cell_id == row.cell_id) & (ref.measured_end_datenum < row.measured_start_datenum)]
    later = ref[(ref.cell_id == row.cell_id) & (ref.measured_start_datenum > row.measured_end_datenum)]
    p = prior.loc[prior.measured_end_datenum.idxmax()] if len(prior) else None
    f = later.loc[later.measured_start_datenum.idxmin()] if len(later) else None
    links.append({"run_uid": row.run_uid, "cell_id": row.cell_id, "operation_index": row.operation_index,
                  "cycle": row.cycle, "start_datenum": row.measured_start_datenum, "end_datenum": row.measured_end_datenum,
                  "previous_reference_uid": p.run_uid if p is not None else None,
                  "previous_reference_age_hours": (row.measured_start_datenum - p.measured_end_datenum) * 24 if p is not None else None,
                  "next_reference_uid": f.run_uid if f is not None else None,
                  "next_reference_delay_after_charge_hours": (f.measured_end_datenum - row.measured_end_datenum) * 24 if f is not None else None,
                  "same_operation_future_reference": bool(f.operation_index == row.operation_index) if f is not None else False,
                  "has_earlier_partial_charge_same_cell": bool(((pc.cell_id == row.cell_id) & (pc.measured_end_datenum < row.measured_start_datenum)).any()),
                  "current_cycle_true_full_capacity_assigned": False, "temperature_channel_present": False,
                  "actual_received_at": "UNKNOWN"})
pd.DataFrame(links).to_csv(O / "partial_charge_reference_timing_links.csv", index=False)
print("step runs", len(st), "reference candidates", len(ref), "partial charges", len(pc))
