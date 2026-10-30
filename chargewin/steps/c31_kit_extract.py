"""KIT: regular-operation charge windows from the 30-s logs and check-up capacity labels."""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

LOGDIR, DOCS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
T0 = 1665593100
GRID = np.arange(0, 601, 30)
cfgz, eocz = zipfile.ZipFile(DOCS / "cfg.zip"), zipfile.ZipFile(DOCS / "cell_eocv2.zip")
cfg = pd.concat([pd.read_csv(io.BytesIO(cfgz.read(n)), sep=";").assign(key=n.split("cell_cfg_")[1][:6]) for n in cfgz.namelist() if n.startswith("cell_cfg_")])
cfg = cfg.drop_duplicates("key").set_index("key")


def process(f):
    win_rows, lab_rows = [], []
    key = f.name.split("30s_")[1][:6]
    c = cfg.loc[key]
    if int(c.age_type) == 1:
        return win_rows, lab_rows
    eoc_name = f"cell_eocv2_{f.name.split('30s_')[1]}"
    e = pd.read_csv(io.BytesIO(eocz.read(eoc_name)), sep=";")
    cu = e[e.cyc_condition == 2]
    cu_iv = np.c_[cu.timestamp_s - cu.cyc_duration_s.fillna(0) - T0 - 600, cu.timestamp_s - T0 + 600]
    labs = cu[(cu.cyc_charged == 0) & (cu.v_min_target_V.round(2) == 2.5) & (cu.i_dischg_min_A.round(1) == -1.0)]
    for _, r in labs.iterrows():
        lab_rows.append(dict(cell=key, t_lab=float(r.timestamp_s - T0), Q=float(-r.delta_q_dischg_Ah)))
    d = pd.read_csv(f, sep=";", usecols=["timestamp_s", "v_raw_V", "i_raw_A", "t_cell_degC"])
    t, v, i, tc = (d[k].values for k in ("timestamp_s", "v_raw_V", "i_raw_A", "t_cell_degC"))
    on = i > 0.05
    edges = np.flatnonzero(np.diff(on.astype(int)))
    starts, ends = list(edges[~on[edges]] + 1), list(edges[on[edges]] + 1)
    if on[0]:
        starts = [0] + starts
    for s0 in starts:
        e_idx = next((x for x in ends if x > s0), len(t))
        ts, te = t[s0], t[e_idx - 1]
        if te - ts < 600 or np.any((cu_iv[:, 0] <= te) & (cu_iv[:, 1] >= ts)):
            continue
        seg = slice(s0, e_idx)
        tt = t[seg] - ts
        q = np.concatenate([[0], np.cumsum(np.diff(tt) * 0.5 * (i[seg][1:] + i[seg][:-1]) / 3600)])
        row = dict(cell=key, t=float(ts), dur=float(te - ts), v_pre=float(v[max(s0 - 1, 0)]),
                   age_type=int(c.age_type), vmax=float(c.V_max_cyc_V), vmin=float(c.V_min_cyc_V),
                   chg=float(c.age_chg_rate), dis=float(c.age_dischg_rate), temp=float(c.age_temp),
                   profile=int(c.age_profile))
        row.update({f"V{g}": x for g, x in zip(GRID, np.interp(GRID, tt, v[seg]))})
        row.update({f"I{g}": x for g, x in zip(GRID, np.interp(GRID, tt, i[seg]))})
        row.update({f"T{g}": x for g, x in zip(GRID, np.interp(GRID, tt, tc[seg]))})
        row["Q600"] = float(np.interp(600, tt, q))
        win_rows.append(row)
    return win_rows, lab_rows


if __name__ == "__main__":
    from multiprocessing import Pool
    files = sorted(LOGDIR.glob("cell_log_age_30s_P*.csv"))
    with Pool(18) as pool:
        res = pool.map(process, files, chunksize=1)
    W = [r for a_, _ in res for r in a_]
    L = [r for _, b_ in res for r in b_]
    OUT.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(W).to_csv(OUT / "kit_windows.csv.gz", index=False)
    pd.DataFrame(L).to_csv(OUT / "kit_checkup_labels.csv", index=False)
    print("done", len(W), len(L))
