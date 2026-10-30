"""UConn: charge windows (first 600 s, 10-s grid) and reference-test capacities, streamed from the original zips."""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SRC, OUT, WHAT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
OUT.mkdir(parents=True, exist_ok=True)
WIN_S, DT = 600, 10
GRID = np.arange(0, WIN_S + DT, DT)


def seg_table(d):
    d = d.rename(columns={"Week Number": "wk", "Cycle Number": "cyc", "Time (s)": "t", "Voltage (V)": "v",
                          "Current (A)": "i", "Capacity (Ah)": "q", "Date (yyyy.mm.dd hh.mm.ss)": "date"})
    d["seg"] = ((d.State != d.State.shift()) | (d.wk != d.wk.shift())).cumsum()
    return d


def cycling(cell_parts, cell):
    rows, carry = [], None
    for name in cell_parts:
        d = seg_table(pd.read_csv(ZIP.open(name), usecols=["Week Number", "Date (yyyy.mm.dd hh.mm.ss)", "Cycle Number",
                                                         "State", "Time (s)", "Voltage (V)", "Current (A)", "Capacity (Ah)"]))
        segs = list(d.groupby("seg", sort=True))
        for k, (_, s) in enumerate(segs):
            st = s.State.iloc[0]
            if st == "CC DChg":
                carry = dict(dis_dq=float(s.q.iloc[-1] - s.q.iloc[0]), dis_v_end=float(s.v.iloc[-1]),
                             dis_i=float(s.i.mean()), dis_end_date=s.date.iloc[-1])
            elif st == "CCCV Chg":
                t = (s.t - s.t.iloc[0]).values
                if len(t) < 5 or np.any(np.diff(t) < 0) or t[-1] < WIN_S:
                    continue
                q = (s.q - s.q.iloc[0]).values
                row = dict(cell=cell, wk=float(s.wk.iloc[0]), cyc=int(s.cyc.iloc[0]), date=s.date.iloc[0],
                           v0=float(s.v.iloc[0]), dq_total=float(q[-1]), dur=float(t[-1]),
                           i_cc=float(s.i.iloc[: max(3, len(s) // 10)].median()))
                if carry:
                    row.update(carry)
                for nm, arr in (("V", s.v.values), ("I", s.i.values), ("Q", q)):
                    row.update({f"{nm}{g}": x for g, x in zip(GRID, np.interp(GRID, t, arr))})
                rows.append(row)
        print(cell, name.rsplit("/", 1)[-1], len(rows), flush=True)
    return rows


def rpt(cell_parts, cell):
    rows = []
    for name in cell_parts:
        d = pd.read_csv(ZIP.open(name), usecols=["Week Number", "Date (yyyy.mm.dd hh.mm.ss)", "Segment Key",
                                                "Time (s)", "Current (A)", "Capacity (Ah)", "Voltage (V)"])
        r = d[d["Segment Key"] == "ref_dchg"]
        for wk, s in r.groupby("Week Number"):
            rows.append(dict(cell=cell, wk=float(wk), date_start=s.iloc[0, 1], date_end=s.iloc[-1, 1],
                             Q_ref=float(s["Capacity (Ah)"].max() - s["Capacity (Ah)"].min()),
                             i_mean=float(s["Current (A)"].mean()), v_end=float(s["Voltage (V)"].iloc[-1]),
                             n=len(s)))
        print(cell, name.rsplit("/", 1)[-1], len(rows), flush=True)
    return rows


ZIP = zipfile.ZipFile(SRC / ("cycling_data.zip" if WHAT == "cycling" else "rpt_data.zip"))
names = sorted(ZIP.namelist(), key=lambda n: (n.split("cell_")[1][:2], int(n.split("part")[1].split(".")[0])) if "cell_" in n else ("", 0))
cells = sorted({n.split("cell_")[1][:2] for n in names if "cell_" in n})
allrows = []
for c in cells:
    parts = [n for n in names if f"cell_{c}_" in n]
    allrows += (cycling if WHAT == "cycling" else rpt)(parts, c)
pd.DataFrame(allrows).to_csv(OUT / f"uconn_{WHAT}_table.csv.gz", index=False)
print("done", len(allrows))
