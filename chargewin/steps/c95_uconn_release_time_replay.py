"""UConn fusion with tests released after the last pulse block and windows after 600 s."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=False)
src = (Path(__file__).with_name("c91_uconn_causal_nested_fusion.py")).read_text(encoding="utf-8")
ns = {"__name__": "c95"}
sys.argv = ["c91", str(ROOT), str(OUT / "_unused")]
exec(compile(src[:src.index("day = ((d.t - d.t.min())")].replace("OUT.mkdir(parents=True, exist_ok=False)", ""),
             "c91_data_banks", "exec"), ns)
d = ns["d"]

P = pd.read_csv(ROOT / "results/C19_UCONN_PULSE_2026-10-23_v1/pulse_resistance.csv.gz")
P["pt"] = pd.to_datetime(P.date, format="%Y.%m.%d %H.%M.%S", errors="coerce")
if P.pt.isna().mean() > 0.5:
    P["pt"] = pd.to_datetime(P.date, errors="coerce")
rel = (P.groupby(["cell", "wk"]).pt.max() + pd.Timedelta(seconds=60)).rename("t_release").reset_index()
d = d.merge(rel, left_on=["cell", "rpt_wk"], right_on=["cell", "wk"], how="left", suffixes=("", "_rel"))
assert d.t_release.notna().all() and len(d) == 12900
t_end = d.t + pd.Timedelta(seconds=600)
assert (t_end <= d.t_next).all(), "a window paired with a check-up ends after that check-up starts"
assert (d.t_release > d.t_next).all()
t0 = d.t.min()
day = ((t_end - t0).dt.total_seconds() / 86400).values
tday = ((d.t_release - t0).dt.total_seconds() / 86400).values
ns["d"] = d
ns.update(day=day, tday=tday)
y, g = ns["y"], ns["g"]
ns["cell_rows"] = {c: np.where(d.cell.values == c)[0] for c in np.unique(d.cell)}
ns["cells_by_group"] = {k: list(np.unique(d.cell[g == k])) for k in ns["groups"]}
ns["slope"] = {c: np.polyfit(day[ix], y[ix], 1)[0] for c, ix in ns["cell_rows"].items()}
exec(compile(src[src.index("def events("):src.index("variants = {")], "c91_funcs", "exec"), ns)
nest = ns["nest"]
pg, pc = ns["evaluate"](lambda k: nest["W"][k], lambda k: nest["WR"][k], "causal_nested_release")

gaps, prefirst = {}, {}
for T in ns["TS"]:
    gl, pf = [], 0
    for c in ns["cell_rows"]:
        ev, _ = ns["events"](c, d.L_W.values, d.L_WR.values, T)
        tt = [t for t, kind, _ in ev if kind == 0]
        gl += list(np.diff(tt))
        pf += sum(1 for t, kind, _ in ev if kind == 1 and t <= tt[0])
    gaps[T] = dict(median=round(float(np.median(gl)), 2), min=round(float(np.min(gl)), 2), max=round(float(np.max(gl)), 2))
    prefirst[T] = pf

c91 = pd.read_csv(ROOT / "results/C91_UCONN_CAUSAL_FUSION_2026-10-29_v1/per_group.csv")
c91 = c91[c91.variant == "causal_nested"]
summ = {}
for T in ns["TS"]:
    x, c, o = pg[pg["T"] == T], pc[pc["T"] == T], c91[c91["T"] == T]
    s = {m: round(float(x[m].mean()), 4) for m in ("ref_drift", "kf_scalar", "kf_bias", "window")}
    s.update(change_vs_C91={m: round(float(x[m].mean() - o[m].mean()), 4) for m in ("ref_drift", "kf_scalar", "kf_bias", "window")},
             bias_beats_ref_groups=int((x.kf_bias < x.ref_drift).sum()),
             C91_bias_beats_ref_groups=int((o.kf_bias < o.ref_drift).sum()),
             ci_bias_minus_ref=ns["boot_ci"](c, "kf_bias", "ref_drift"),
             ci_scalar_minus_ref=ns["boot_ci"](c, "kf_scalar", "ref_drift"),
             gate_on_groups=int(x.gate_on.sum()), gate_correct_groups=int(x.gate_correct.sum()),
             realized_gap_days=gaps[T], windows_before_first_test=prefirst[T], windows_total=len(d))
    summ[f"T{T}"] = s
maxchg = max(abs(v) for s in summ.values() for v in s["change_vs_C91"].values())
wins_ok = all(abs(s["bias_beats_ref_groups"] - s["C91_bias_beats_ref_groups"]) <= 1 for s in summ.values())
summ["decision"] = dict(max_abs_change_Ah=round(maxchg, 4), wins_within_1=wins_ok,
                        keep_C91_numbers=bool(maxchg <= 0.002 and wins_ok))
pg.to_csv(OUT / "per_group.csv", index=False)
pc.to_csv(OUT / "per_cell.csv", index=False)
json.dump(summ, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(summ, indent=1))
