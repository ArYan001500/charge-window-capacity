"""KIT: source-only perceptron window sensor."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

W_DIR, C32, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=False)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
p = pd.read_csv(C32 / "window_predictions.csv.gz")
w = pd.read_csv(W_DIR / "kit_windows.csv.gz")
d = p.merge(w, on=["cell", "t"], how="left")
d["dVa"], d["dVb"] = d.V60 - d.V0, d.V600 - d.V300
d["Tmean"] = d[[f"T{s}" for s in range(0, 601, 30)]].mean(1)
F = [f"V{s}" for s in range(0, 601, 60)] + ["Q600", "dVa", "dVb", "I0", "I300", "I600", "Tmean"]


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


d["pred"] = np.nan
for g in sorted(d.group.unique()):
    src = d[(d.group != g) & d.near]
    tm = d.group == g
    mu, sd = src[F].values.mean(0), src[F].values.std(0) + 1e-9
    ym, ysd = src.Q.mean(), src.Q.std()
    Xs, Xt = (src[F].values - mu) / sd, (d.loc[tm, F].values - mu) / sd
    d.loc[tm, "pred"] = np.mean([train(Xs, (src.Q.values - ym) / ysd, Xt) for _ in range(3)], axis=0) * ysd + ym
    print("group", g, flush=True)
d[["cell", "group", "t", "near", "Q", "pred"]].to_csv(OUT / "window_predictions.csv.gz", index=False)
near = d[d.near]
print("passive MLP MAE (near rows, mean over groups):",
      near.groupby("group").apply(lambda z: (z.pred - z.Q).abs().groupby(z.cell).mean().mean()).mean())
