"""Group-max intervals around fused estimates."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import c54_hust_kalman
from c54_hust_kalman import prep as hust_prep

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
R = ROOT / "results"
c54_hust_kalman.C14 = R / "C14_HUST_LOGO_2026-10-23_v1"
EPS = 1e-3
GRID = [(qn, qb) for qn in (1e-5, 1e-4, 1e-3) for qb in (1e-6, 1e-5, 1e-4, 1e-3)]


def kf(ev, drift, qn, qb, r_test, r_win, p_b0):
    """ev: list of (day, is_test, z, row) in time order. Returns {row: (est, sd)} for window rows."""
    s, out, prev = None, {}, None
    for day, is_test, z, row in ev:
        if s is not None:
            dt = day - prev
            s = s + np.array([drift * dt, 0.0])
            P = P + np.diag([qn * dt, qb * dt])
        prev = day
        if is_test:
            if s is None:
                s, P = np.array([z, 0.0]), np.diag([r_test, p_b0])
            else:
                H = np.array([1.0, 0.0])
                K = P @ H / (H @ P @ H + r_test)
                s, P = s + K * (z - H @ s), P - np.outer(K, H @ P)
        elif s is not None:
            H = np.array([1.0, 1.0])
            K = P @ H / (H @ P @ H + r_win)
            s, P = s + K * (z - H @ s), P - np.outer(K, H @ P)
            out[row] = (s[0], np.sqrt(max(P[0, 0], 0.0)))
    return out


def build(name):
    """Returns d (rows: cell, group, Q, pred, score_row), events per cell, r_test (None = estimate), slopes per cell."""
    if name == "UConn":
        c19 = sorted(R.glob("C19_*"))[0]
        d = pd.read_csv(c19 / "scored.csv.gz", parse_dates=["t"]).sort_values(["cell", "t"]).reset_index(drop=True)
        d["day"] = (d.t - d.t.min()).dt.total_seconds() / 86400
        d = d.rename(columns={"W": "pred"})
        ev, keep = {}, np.zeros(len(d), bool)
        for c, x in d.groupby("cell"):
            e, last = [], None
            for i, day, w, wt in zip(x.index, x.day, x.pred, x["W+Rall"]):
                if last is None or day - last >= 14:
                    e.append((day, True, wt, i))
                    last = day
                else:
                    e.append((day, False, w, i))
                    keep[i] = True
            ev[c] = e
        d["score_row"] = keep
        slopes = {c: np.polyfit(x.day, x.Q, 1)[0] for c, x in d.groupby("cell")}
        return d, ev, None, slopes, "W+Rall"
    if name == "KIT":
        d = pd.read_csv(R / "C32_KIT_KALMAN_2026-10-24_v1/window_predictions.csv.gz").sort_values(["cell", "t"]).reset_index(drop=True)
        lab = pd.read_csv(ROOT / "tables/C31_KIT_2026-10-24_v1/kit_checkup_labels.csv")
        lab = lab[lab.cell.isin(d.cell.unique())]
        labs = {c: (x.t_lab.values / 86400, x.Q.values) for c, x in lab.sort_values("t_lab").groupby("cell")}
    else:
        d, lb = hust_prep("random")
        labs = {c: (x.t_lab.values / 86400, x.Q.values) for c, x in lb.items()}
        d = d[d.near].reset_index(drop=True)
    d["day"] = d.t / 86400
    d["score_row"] = d.near.astype(bool)
    ev = {}
    for c, x in d.groupby("cell"):
        e = [(t, False, p, i) for t, p, i in zip(x.day, x.pred, x.index)]
        if c in labs:
            e += [(t, True, q, -1) for t, q in zip(*labs[c])]
        ev[c] = sorted(e, key=lambda z: (z[0], not z[1]))
    slopes = {c: np.polyfit(t, q, 1)[0] for c, (t, q) in labs.items() if len(t) >= 2}
    return d, ev, 1e-4, slopes, None


def fold(args):
    name, g = args
    d, ev, r_test, slopes, test_col = DATA[name]
    cg = d.groupby("cell").group.first()
    src = list(cg[cg != g].index)
    tgt = list(cg[cg == g].index)
    s = d[d.cell.isin(src) & d.score_row]
    drift = float(np.mean([slopes[c] for c in src if c in slopes]))
    if r_test is None:
        ss = d[d.cell.isin(src)]
        r_test = float(np.var(ss[test_col] - ss.Q))
    bias = (s.pred - s.Q).groupby(s.cell).mean()
    p_b0 = float(np.var(bias) + np.mean(bias) ** 2)
    r_win = float(np.var((s.pred - s.Q) - s.cell.map(bias)))

    def run(cells, qn, qb):
        o = {}
        for c in cells:
            o.update(kf(ev[c], drift, qn, qb, r_test, r_win, p_b0))
        return o

    def eqmae(o):
        x = d.loc[[k for k in o if d.at[k, "score_row"]]]
        e = pd.Series({k: o[k][0] for k in x.index})
        return float((e - x.Q).abs().groupby(x.cell).mean().mean())

    best = min(GRID, key=lambda p: eqmae(run(src, *p)))
    o_src, o_tgt = run(src, *best), run(tgt, *best)
    pack = lambda o, role: [dict(row=k, est=v[0], sd=v[1], role=role) for k, v in o.items() if d.at[k, "score_row"]]
    return g, pd.DataFrame(pack(o_src, "src") + pack(o_tgt, "tgt"))


DATA = {}
for _n in ("UConn", "KIT", "HUST"):
    DATA[_n] = build(_n)

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=False)
    jobs = [(n, g) for n in DATA for g in sorted(DATA[n][0].group.unique())]
    with Pool(18) as pool:
        outs = pool.map(fold, jobs)
    rows, maes = [], {}
    for (name, _), (g, f) in zip(jobs, outs):
        d = DATA[name][0]
        f = f.assign(Q=d.loc[f.row, "Q"].values, cell=d.loc[f.row, "cell"].values, grp=d.loc[f.row, "group"].values,
                     pred=d.loc[f.row, "pred"].values)
        f["r"], f["sc"] = (f.est - f.Q).abs(), f.sd + EPS
        f["z"] = f.r / f.sc
        s, x = f[f.role == "src"], f[f.role == "tgt"]
        qa, qb = np.quantile(s.r, 0.9), s.groupby("grp").r.quantile(0.9).max()
        qc, qd = np.quantile(s.z, 0.9), s.groupby("grp").z.quantile(0.9).max()
        for m, half in (("A_pooled", np.full(len(x), qa)), ("B_groupmax", np.full(len(x), qb)),
                        ("C_norm", qc * x.sc.values), ("D_norm_groupmax", qd * x.sc.values)):
            rows.append(dict(dataset=name, method=m, group=g, coverage=float((x.r.values <= half).mean()), width=float(2 * half.mean())))
        maes.setdefault(name, []).append(dict(kf=float(x.r.groupby(x.cell).mean().mean()),
                                              passive=float((x.pred - x.Q).abs().groupby(x.cell).mean().mean())))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "per_group.csv", index=False)
    LIM = {"UConn": 0.291, "HUST": 0.109, "KIT": 0.385}
    res = {"MAE_on_scored_rows": {k: {kk: round(float(np.mean([z[kk] for z in v])), 4) for kk in ("kf", "passive")} for k, v in maes.items()}}
    for (ds, m), x in r.groupby(["dataset", "method"]):
        res.setdefault(m, {})[ds] = dict(mean_cov=round(x.coverage.mean(), 3), min_cov=round(x.coverage.min(), 3),
                                         groups_below_80=int((x.coverage < 0.8).sum()), mean_width_Ah=round(x.width.mean(), 4),
                                         width_ok=bool(x.width.mean() <= LIM[ds]))
    for m in ("A_pooled", "B_groupmax", "C_norm", "D_norm_groupmax"):
        v = res[m]
        v["PASS"] = all(z["mean_cov"] >= 0.88 and z["min_cov"] >= 0.80 and z["width_ok"] for z in v.values())
    json.dump(res, open(OUT / "summary.json", "w"), indent=2)
    print(json.dumps(res, indent=1))
