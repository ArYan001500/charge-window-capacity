"""Adapted Tao and See indicator on UConn and HUST."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=False)
R = ROOT / "results"
TS = list(range(0, 601, 10))
dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(87)
CAND = {"UConn": [(3.60, 3.70), (3.70, 3.80), (3.80, 3.90), (3.90, 4.00), (4.00, 4.10)],
        "HUST_masked": [(3.30, 3.40), (3.40, 3.50), (3.50, 3.60)],
        "HUST_fixed": [(3.30, 3.40), (3.40, 3.50), (3.50, 3.60)]}


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Zs = (X - mu) / sd
    w = np.linalg.solve(Zs.T @ Zs + alpha * np.eye(Zs.shape[1]), Zs.T @ (y - y.mean()))
    return lambda Xn: (Xn - mu) / sd @ w + y.mean()


def pick_alpha(X, y, g):
    return min(np.logspace(-3, 4, 15), key=lambda a: np.mean(
        [np.abs(ridge_fit(X[g != k], y[g != k], a)(X[g == k]) - y[g == k]).mean() for k in np.unique(g)]))


def mlp_fit(X, y, seed=0):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=dev)
    ym, ys = y.mean(), y.std() + 1e-12
    yt = torch.tensor((y - ym) / ys, dtype=torch.float32, device=dev)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    for _ in range(300):
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            loss = ((net(Xt[b]).squeeze(1) - yt[b]) ** 2).mean()
            loss.backward()
            opt.step()
    net.eval()
    return lambda Xn: net(torch.tensor((Xn - mu) / sd, dtype=torch.float32, device=dev)).squeeze(1).detach().cpu().numpy() * ys + ym


def qv_features(d, va, vb, n=16, dt=10):
    V = d[[f"V{s}" for s in TS]].values
    I = d[[f"I{s}" for s in TS]].values
    Q = np.concatenate([np.zeros((len(d), 1)), np.cumsum((I[:, 1:] + I[:, :-1]) / 2 * dt / 3600, axis=1)], axis=1)
    Vm = np.maximum.accumulate(V, axis=1)
    ok = (Vm[:, 0] <= va) & (Vm[:, -1] >= vb)
    grid = np.linspace(va, vb, n)
    F = np.full((len(d), n), np.nan)
    for i in np.where(ok)[0]:
        v, q = Vm[i], Q[i]
        keep = np.r_[True, np.diff(v) > 1e-6]
        F[i] = np.interp(grid, v[keep], q[keep])
    F = F - F[:, [0]]
    return ok, F


def eq_mae(df, col):
    return float(df.groupby("group").apply(lambda z: (z[col] - z.Q).abs().groupby(z.cell).mean().mean()).mean())


def paired_ci(df, a, b):
    t = df.assign(da=(df[a] - df.Q).abs(), db=(df[b] - df.Q).abs()).groupby(["group", "cell"])[["da", "db"]].mean()
    t["d"] = t.da - t.db
    per = [g.d.values for _, g in t.groupby(level=0)]
    boot = np.array([np.mean([p[rng.integers(0, len(p), len(p))].mean() for p in per]) for _ in range(2000)])
    pg = t.groupby(level=0)[["da", "db"]].mean()
    return dict(point=round(float(np.mean([p.mean() for p in per])), 4),
                CI95=[round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
                ours_better_groups=f"{int((pg.da < pg.db).sum())}/{len(pg)}")


def load_uconn():
    s = pd.read_csv(R / "C11_UCONN_LOGO_2026-10-23_v1/scored_logo.csv.gz", parse_dates=["t"])
    cyc = pd.read_csv(ROOT / "tables/C10_UCONN_2026-10-23_v1/uconn_cycling_table.csv.gz",
                      usecols=["cell", "date"] + [f"V{t}" for t in TS] + [f"I{t}" for t in TS])
    cyc["t"] = pd.to_datetime(cyc.date, format="%Y.%m.%d %H.%M.%S", errors="coerce")
    if cyc.t.isna().mean() > 0.5:
        cyc["t"] = pd.to_datetime(cyc.date, errors="coerce")
    cyc = cyc.dropna(subset=["t"]).drop_duplicates(["cell", "t"])
    d = s.merge(cyc.drop(columns="date"), on=["cell", "t"], how="left")
    assert len(d) == len(s) and d.V600.notna().all(), "UConn join failed"
    return d


data = {"UConn": load_uconn(),
        "HUST_masked": pd.read_csv(R / "C14_HUST_LOGO_2026-10-23_v1/scored_random.csv.gz"),
        "HUST_fixed": pd.read_csv(R / "C14_HUST_LOGO_2026-10-23_v1/scored_fixed.csv.gz")}
res, preds = {}, []
for name, d in data.items():
    assert abs(eq_mae(d, "MW_quad") - {"UConn": 0.0819, "HUST_masked": 0.0459, "HUST_fixed": 0.0131}[name]) < 6e-5
    out = {}
    for va, vb in CAND[name]:
        ok, F = qv_features(d, va, vb)
        x = d[ok].copy().reset_index(drop=True)
        F = F[ok]
        key = f"{va:.2f}-{vb:.2f}V"
        r = dict(coverage=round(float(ok.mean()), 3), n=int(ok.sum()), groups=int(x.group.nunique()), cells=int(x.cell.nunique()))
        if x.group.nunique() >= 4 and len(x) >= 200:
            x["TS_ridge"], x["TS_mlp"] = np.nan, np.nan
            for g in sorted(x.group.unique()):
                tr, te = x.group != g, x.group == g
                Xs, ys, gs = F[tr], x.Q.values[tr], x.group.values[tr]
                x.loc[te, "TS_ridge"] = ridge_fit(Xs, ys, pick_alpha(Xs, ys, gs))(F[te])
                x.loc[te, "TS_mlp"] = mlp_fit(Xs, ys)(F[te])
            r.update({c: round(eq_mae(x, c), 4) for c in ("MW_quad", "TS_ridge", "TS_mlp")})
            r["ours_minus_TS_mlp"] = paired_ci(x, "MW_quad", "TS_mlp")
            r["ours_minus_TS_ridge"] = paired_ci(x, "MW_quad", "TS_ridge")
            r["evaluable"] = True
            preds.append(x[["cell", "group", "Q", "MW_quad", "TS_ridge", "TS_mlp"]].assign(dataset=name, window=key))
        else:
            r["evaluable"] = False
        out[key] = r
        print(name, key, json.dumps(r), flush=True)
    ev = [k for k, v in out.items() if v["evaluable"]]
    out["primary"] = max(ev, key=lambda k: out[k]["coverage"]) if ev else "NONE_EVALUABLE"
    res[name] = out
pd.concat(preds).to_csv(OUT / "predictions.csv.gz", index=False)
for crit, name in (("T1_UConn", "UConn"), ("T2_HUST_masked", "HUST_masked")):
    p = res[name]["primary"]
    res[crit + "_PASS"] = bool(p != "NONE_EVALUABLE" and res[name][p]["ours_minus_TS_mlp"]["CI95"][1] < 0)
json.dump(res, open(OUT / "summary.json", "w"), indent=2)
print({k: v for k, v in res.items() if k.endswith("PASS")})
