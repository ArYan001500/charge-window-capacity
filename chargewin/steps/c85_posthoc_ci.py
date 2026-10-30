"""Cluster-bootstrap confidence intervals for the competitor comparisons."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=False)
R = ROOT / "results"
TS = list(range(0, 601, 30))
WINDOWS = [(3.60, 3.70), (3.70, 3.80), (3.80, 3.90), (3.90, 4.00)]
dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(85)


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


def qv_features(d, va, vb, n=16):
    V = d[[f"V{s}" for s in TS]].values
    I = d[[f"I{s}" for s in TS]].values
    Q = np.concatenate([np.zeros((len(d), 1)), np.cumsum((I[:, 1:] + I[:, :-1]) / 2 * 30 / 3600, axis=1)], axis=1)
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
    """Point and 95% cluster-bootstrap CI of eqMAE(a) - eqMAE(b), cells resampled within groups."""
    t = df.assign(da=(df[a] - df.Q).abs(), db=(df[b] - df.Q).abs()).groupby(["group", "cell"])[["da", "db"]].mean()
    t["d"] = t.da - t.db
    per = [g.d.values for _, g in t.groupby(level=0)]
    point = float(np.mean([p.mean() for p in per]))
    boot = np.array([np.mean([p[rng.integers(0, len(p), len(p))].mean() for p in per]) for _ in range(2000)])
    pg = t.groupby(level=0)[["da", "db"]].mean()
    return dict(point=round(point, 4), CI95=[round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
                a_better_groups=f"{int((pg.da < pg.db).sum())}/{len(pg)}", mae_a=round(eq_mae(df, a), 4), mae_b=round(eq_mae(df, b), 4))


res = {"A_taosee": {}, "B_mlp_vs_ridge": {}}
c55 = json.load(open(R / "C55_KIT_TAOSEE_BASELINE_2026-10-24_v1/summary.json"))
s = pd.read_csv(R / "C31_KIT_LOGO_2026-10-24_usage/scored.csv.gz")
w = pd.read_csv(ROOT / "tables/C31_KIT_2026-10-24_v1/kit_windows.csv.gz", usecols=["cell", "t"] + [f"V{t}" for t in TS] + [f"I{t}" for t in TS])
d = s.merge(w, on=["cell", "t"], how="left")
preds = []
for va, vb in WINDOWS:
    ok, F = qv_features(d, va, vb)
    x = d[ok].copy().reset_index(drop=True)
    F = F[ok]
    key = f"{va:.2f}-{vb:.2f}V"
    x["TS_ridge"], x["TS_mlp"] = np.nan, np.nan
    for g in sorted(x.group.unique()):
        tr, te = x.group != g, x.group == g
        Xs, ys, gs = F[tr], x.Q.values[tr], x.group.values[tr]
        x.loc[te, "TS_ridge"] = ridge_fit(Xs, ys, pick_alpha(Xs, ys, gs))(F[te])
        x.loc[te, "TS_mlp"] = mlp_fit(Xs, ys)(F[te])
    rep = {c: round(eq_mae(x, c), 4) for c in ("TS_ridge", "TS_mlp", "MW_quadT")}
    ref = {c: c55[key][c] for c in rep}
    r = dict(n=len(x), reproduced=rep, c55=ref,
             ridge_and_ours_exact=bool(abs(rep["TS_ridge"] - ref["TS_ridge"]) < 1e-4 and abs(rep["MW_quadT"] - ref["MW_quadT"]) < 1e-4),
             mlp_abs_diff=round(abs(rep["TS_mlp"] - ref["TS_mlp"]), 4),
             ours_minus_TS_mlp=paired_ci(x, "MW_quadT", "TS_mlp"), ours_minus_TS_ridge=paired_ci(x, "MW_quadT", "TS_ridge"))
    res["A_taosee"][key] = r
    preds.append(x[["cell", "group", "t", "Q", "MW_quadT", "TS_ridge", "TS_mlp"]].assign(window=key))
    print(key, json.dumps(r), flush=True)
pd.concat(preds).to_csv(OUT / "taosee_predictions.csv.gz", index=False)

k32 = pd.read_csv(R / "C32_KIT_KALMAN_2026-10-24_v1/window_predictions.csv.gz").rename(columns={"pred": "ridge"})
k35 = pd.read_csv(R / "C35_KIT_MLP_PASSIVE_2026-10-24_v1/window_predictions.csv.gz").rename(columns={"pred": "mlp"})
k = k32.merge(k35[["cell", "t", "mlp"]], on=["cell", "t"])
assert len(k) == len(k32) == len(k35)
res["B_mlp_vs_ridge"]["KIT"] = paired_ci(k[k.near], "mlp", "ridge")
u = pd.read_csv(R / "C15_UCONN_DANN_2026-10-23_v1/scored.csv.gz")
res["B_mlp_vs_ridge"]["UConn"] = paired_ci(u, "MLP_src", "MW_quad")
print(json.dumps(res["B_mlp_vs_ridge"], indent=1))
json.dump(res, open(OUT / "summary.json", "w"), indent=2)
