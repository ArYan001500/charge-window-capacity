"""UConn: charge curves of the reference tests on a common voltage grid."""
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RPTZIP, OUT = Path(sys.argv[1]), Path(sys.argv[2])
VGRID = np.round(np.arange(3.0, 4.2001, 0.005), 3)


def one(name):
    z = zipfile.ZipFile(RPTZIP)
    cell = int(name.split("cell_")[1][:2])
    d = pd.read_csv(z.open(name), usecols=["Week Number", "Date (yyyy.mm.dd hh.mm.ss)", "Time (s)", "Voltage (V)",
                                          "Current (A)", "Segment Key"])
    rows = []
    for wk, s in d[d["Segment Key"] == "ref_chg"].groupby("Week Number"):
        s = s[s["Current (A)"] > 0.7]
        if len(s) < 100:
            continue
        t = s["Time (s)"].values
        i = s["Current (A)"].values
        q = np.concatenate([[0], np.cumsum(np.diff(t) * 0.5 * (i[1:] + i[:-1]) / 3600)])
        v = np.maximum.accumulate(s["Voltage (V)"].values)
        keep = np.r_[True, np.diff(v) > 0]
        qv = np.interp(VGRID, v[keep], q[keep], left=np.nan, right=np.nan)
        rows.append(dict(cell=cell, wk=float(wk), date=s.iloc[0, 1], **{f"q{g:.3f}": x for g, x in zip(VGRID, qv)}))
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=False)
    names = sorted(n for n in zipfile.ZipFile(RPTZIP).namelist() if n.endswith(".csv"))
    with Pool(18) as p:
        res = p.map(one, names, chunksize=1)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(OUT / "uconn_ref_chg_QofV.csv.gz", index=False)
    print("done", df.shape)
