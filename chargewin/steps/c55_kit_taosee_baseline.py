"""KIT: adapted Tao and See indicator on the fixed voltage window."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=False)
TS = list(range(0, 601, 30))
WINDOWS = [(3.60, 3.70), (3.70, 3.80), (3.80, 3.90), (3.90, 4.00), (4.00, 4.10)]
dev = "cuda" if torch.cuda.is_available() else "cpu"


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


s = pd.read_csv(ROOT / "results/C31_KIT_LOGO_2026-10-24_usage/scored.csv.gz")
w = pd.read_csv(ROOT / "tables/C31_KIT_2026-10-24_v1/kit_windows.csv.gz", usecols=["cell", "t"] + [f"V{t}" for t in TS] + [f"I{t}" for t in TS])
d = s.merge(w, on=["cell", "t"], how="left")
assert d[f"V{TS[-1]}"].notna().all()
res = {"n_rows_C31": len(d)}
for va, vb in WINDOWS:
    ok, F = qv_features(d, va, vb)
    x = d[ok].copy().reset_index(drop=True)
    F = F[ok]
    key = f"{va:.2f}-{vb:.2f}V"
    r = dict(coverage=round(float(ok.mean()), 3), n=int(ok.sum()), groups=int(x.group.nunique()), cells=int(x.cell.nunique()))
    if x.group.nunique() >= 4 and len(x) >= 200:
        x["TS_ridge"], x["TS_mlp"] = np.nan, np.nan
        for g in sorted(x.group.unique()):
            tr, te = x.group != g, x.group == g
            if te.sum() == 0:
                continue
            Xs, ys, gs = F[tr], x.Q.values[tr], x.group.values[tr]
            x.loc[te, "TS_ridge"] = ridge_fit(Xs, ys, pick_alpha(Xs, ys, gs))(F[te])
            x.loc[te, "TS_mlp"] = mlp_fit(Xs, ys)(F[te])
        for col in ("TS_ridge", "TS_mlp", "MW_quad", "MW_quadT", "P1", "M0"):
            r[col] = round(eq_mae(x, col), 4)
        pg = x.groupby("group").apply(lambda z: pd.Series({c: (z[c] - z.Q).abs().groupby(z.cell).mean().mean() for c in ("TS_ridge", "TS_mlp", "MW_quadT")}))
        r["MW_quadT_beats_TS_ridge"] = f"{int((pg.MW_quadT < pg.TS_ridge).sum())}/{len(pg)}"
        r["MW_quadT_beats_TS_mlp"] = f"{int((pg.MW_quadT < pg.TS_mlp).sum())}/{len(pg)}"
    res[key] = r
prim = max((k for k in res if k.endswith("V")), key=lambda k: res[k]["coverage"])
res["primary_window_by_label_free_coverage"] = prim
json.dump(res, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(res, indent=1))
