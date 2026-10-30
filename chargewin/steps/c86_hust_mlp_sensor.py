"""HUST: perceptron window sensor against the ridge sensor."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=False)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
rng = np.random.default_rng(86)
F = [f"V{s}" for s in range(0, 601, 60)] + ["Q600", "dVa", "dVb", "I0", "I300", "I600"]


def train(Xs, ys, Xt, epochs=60, bs=256):
    net = nn.Sequential(nn.Linear(Xs.shape[1], 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xs, ys, Xt = (torch.tensor(a, dtype=torch.float32, device=dev) for a in (Xs, ys, Xt))
    for _ in range(epochs):
        perm = torch.randperm(len(Xs), device=dev)
        for s in range(max(1, len(Xs) // bs)):
            idx = perm[s * bs:(s + 1) * bs]
            loss = ((net(Xs[idx]).squeeze(1) - ys[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return net(Xt).squeeze(1).cpu().numpy()


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
                mlp_better_groups=f"{int((pg.da < pg.db).sum())}/{len(pg)}")


res = {}
for mode in ("random", "fixed"):
    d = pd.read_csv(R / f"C14_HUST_LOGO_2026-10-23_v1/scored_{mode}.csv.gz")
    d["MLP"] = np.nan
    for g in sorted(d.group.unique()):
        src, tm = d[d.group != g], d.group == g
        mu, sd = src[F].values.mean(0), src[F].values.std(0) + 1e-9
        ym, ysd = src.Q.mean(), src.Q.std()
        Xs, Xt = (src[F].values - mu) / sd, (d.loc[tm, F].values - mu) / sd
        d.loc[tm, "MLP"] = np.mean([train(Xs, (src.Q.values - ym) / ysd, Xt) for _ in range(3)], axis=0) * ysd + ym
    ci = paired_ci(d, "MLP", "MW_quad")
    res[mode] = dict(ridge_MW_quad=round(eq_mae(d, "MW_quad"), 4), MLP=round(eq_mae(d, "MLP"), 4), MLP_minus_ridge=ci,
                     PASS=bool(ci["CI95"][1] < 0))
    d[["cell", "group", "cyc", "Q", "MW_quad", "MLP"]].to_csv(OUT / f"predictions_{mode}.csv.gz", index=False)
    print(mode, json.dumps(res[mode]), flush=True)
res["M1_masked_PASS"], res["M2_fixed_PASS"] = res["random"]["PASS"], res["fixed"]["PASS"]
json.dump(res, open(OUT / "summary.json", "w"), indent=2)
