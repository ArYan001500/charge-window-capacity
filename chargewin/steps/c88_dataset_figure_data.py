"""Data for the dataset figure."""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from safe_pickle import safe_loads

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=False)
W, RAW, RES = ROOT / "tables", ROOT / "data", ROOT / "results"
T0 = 1665593100


def resample(q, v, n=120):
    """V on n points of accumulated charge q (monotone q)."""
    keep = np.r_[True, np.diff(q) > 1e-6]
    q, v = q[keep], v[keep]
    g = np.linspace(0, q[-1], n)
    return g, np.interp(g, q, v)


curves, traj, med = [], [], {}

c = pd.read_csv(W / "C38_UCONN_REF_CURVES_2026-10-24_v1/uconn_ref_chg_QofV.csv.gz")
rpt = pd.read_csv(W / "C10_UCONN_2026-10-23_v1/uconn_rpt_table.csv.gz")
qc = [k for k in c.columns if k.startswith("q")]
vv = np.array([float(k[1:]) for k in qc])
x = c[c.cell == 21].merge(rpt[["cell", "wk", "Q_ref"]], on=["cell", "wk"]).dropna(subset=["Q_ref"]).sort_values("wk")
for _, r in x.iterrows():
    q = r[qc].values.astype(float)
    ok = ~np.isnan(q)
    qq = np.maximum.accumulate(q[ok])
    g, v = resample(qq - qq[0], vv[ok])
    curves += [dict(dataset="UConn", age=r.wk * 7, q=gi, v=vi, Q=r.Q_ref) for gi, vi in zip(g, v)]
for cell, z in rpt.dropna(subset=["Q_ref"]).groupby("cell"):
    traj += [dict(dataset="UConn", cell=str(cell), age=w * 7, Q=q) for w, q in zip(z.wk, z.Q_ref)]
cyc = pd.read_csv(W / "C10_UCONN_2026-10-23_v1/uconn_cycling_table.csv.gz", usecols=["Q600"])
med["UConn"] = float(cyc.Q600.median())

zf = zipfile.ZipFile(RAW / "hust/our_data.zip")
names = sorted(n for n in zf.namelist() if n.endswith(".pkl"))
ex_cell = Path(names[0]).stem
for name in names:
    cell = Path(name).stem
    o = safe_loads(zf.read(name))[cell]
    traj += [dict(dataset="HUST", cell=cell, age=float(k), Q=float(q) / 1000) for k, q in sorted(o["dq"].items())[::5]]
    if cell != ex_cell:
        continue
    ks = sorted(o["data"].keys())
    for k in ks[::max(1, len(ks) // 40)]:
        d = o["data"][k]
        ch = d[d.Status.str.contains("charge") & ~d.Status.str.contains("discharge")]
        if len(ch) < 50:
            continue
        t = ch["Time (s)"].values.astype(float)
        i = ch["Current (mA)"].values / 1000
        q = np.concatenate([[0], np.cumsum(np.diff(t - t[0]) * 0.5 * (i[1:] + i[:-1]) / 3600)])
        g, v = resample(q, ch["Voltage (V)"].values)
        curves += [dict(dataset="HUST", age=float(k), q=gi, v=vi, Q=float(o["dq"][k]) / 1000) for gi, vi in zip(g, v)]
hw = pd.read_csv(RES / "C14_HUST_LOGO_2026-10-23_v1/scored_random.csv.gz", usecols=["Q600"])
med["HUST"] = float(hw.Q600.median())

docs = RAW / "kit/result_data"
cfgz = zipfile.ZipFile(docs / "cfg.zip")
cfg = pd.concat([pd.read_csv(io.BytesIO(cfgz.read(n)), sep=";").assign(key=n.split("cell_cfg_")[1][:6])
                 for n in cfgz.namelist() if n.startswith("cell_cfg_")]).drop_duplicates("key").set_index("key")
sel = cfg[(cfg.age_type == 2) & (cfg.V_max_cyc_V.round(2) == 4.2) & (cfg.V_min_cyc_V.round(2) == 2.5)
          & (cfg.age_chg_rate.round(2) == 0.33) & (cfg.age_dischg_rate.round(2) == 1.0) & (cfg.age_temp == 25)]
key = sel.index[0]
lab = pd.read_csv(W / "C31_KIT_2026-10-24_v1/kit_checkup_labels.csv")
f = next((RAW / "kit/cell_log_age_30s").glob(f"cell_log_age_30s_{key}*.csv"))
d = pd.read_csv(f, sep=";", usecols=["timestamp_s", "v_raw_V", "i_raw_A"])
t, v, i = (d[k].values for k in ("timestamp_s", "v_raw_V", "i_raw_A"))
on = i > 0.05
edges = np.flatnonzero(np.diff(on.astype(int)))
starts, ends = list(edges[~on[edges]] + 1), list(edges[on[edges]] + 1)
segs = []
for s0 in starts:
    e = next((x for x in ends if x > s0), len(t))
    if t[e - 1] - t[s0] > 2.5 * 3600 and v[s0] < 2.7:
        segs.append((s0, e))
ll = lab[lab.cell == key].sort_values("t_lab")
for s0, e in segs[::max(1, len(segs) // 40)]:
    tt = t[s0:e] - t[s0]
    q = np.concatenate([[0], np.cumsum(np.diff(tt) * 0.5 * (i[s0:e][1:] + i[s0:e][:-1]) / 3600)])
    g, vv2 = resample(q, v[s0:e])
    Q = float(np.interp(t[s0], ll.t_lab.values, ll.Q.values))
    curves += [dict(dataset="KIT", age=float(t[s0]) / 86400, q=gi, v=vi, Q=Q) for gi, vi in zip(g, vv2)]
for cell, z in lab.groupby("cell"):
    traj += [dict(dataset="KIT", cell=cell, age=tl / 86400, Q=q) for tl, q in zip(z.t_lab, z.Q)]
kw = pd.read_csv(W / "C31_KIT_2026-10-24_v1/kit_windows.csv.gz", usecols=["Q600"])
med["KIT"] = float(kw.Q600.median())

pd.DataFrame(curves).to_csv(OUT / "example_curves.csv.gz", index=False)
pd.DataFrame(traj).to_csv(OUT / "capacity_trajectories.csv.gz", index=False)
pd.Series({"UConn_cell": "21", "HUST_cell": ex_cell, "KIT_cell": key, **{f"median_Q600_{k}": v for k, v in med.items()}}
          ).to_json(OUT / "summary.json", indent=1)
print(ex_cell, key, med, len(curves), len(traj))
