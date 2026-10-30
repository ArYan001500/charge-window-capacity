"""UConn: source-only perceptron and domain-adversarial network on the window inputs."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

C11, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=False)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)

C10 = Path(sys.argv[3])
t11 = pd.read_csv(C11 / "scored_logo.csv.gz", parse_dates=["t"])
cyc = pd.read_csv(C10 / "uconn_cycling_table.csv.gz")
cyc["t"] = pd.to_datetime(cyc.date)
d = t11.merge(cyc, on=["cell", "t"], how="left", suffixes=("", "_c"))
VG = [f"V{s}" for s in range(0, 601, 60)]
d["dVa"], d["dVb"] = d.V60 - d.V0, d.V600 - d.V300
F = VG + ["Q600", "dVa", "dVb", "I0", "I300", "I600"]
d = d.dropna(subset=F)


class GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def make():
    feat = nn.Sequential(nn.Linear(len(F), 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()).to(dev)
    reg = nn.Linear(32, 1).to(dev)
    dom = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1)).to(dev)
    return feat, reg, dom


def train(Xs, ys, Xt, dann, epochs=60, bs=256):
    feat, reg, dom = make()
    params = list(feat.parameters()) + list(reg.parameters()) + (list(dom.parameters()) if dann else [])
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
    Xs, ys, Xt = (torch.tensor(a, dtype=torch.float32, device=dev) for a in (Xs, ys, Xt))
    n, steps = len(Xs), max(1, len(Xs) // bs)
    for ep in range(epochs):
        p = ep / epochs
        lam = 2 / (1 + np.exp(-10 * p)) - 1
        perm = torch.randperm(n, device=dev)
        for s in range(steps):
            idx = perm[s * bs:(s + 1) * bs]
            hs = feat(Xs[idx])
            loss = ((reg(hs).squeeze(1) - ys[idx]) ** 2).mean()
            if dann:
                ti = torch.randint(0, len(Xt), (len(idx),), device=dev)
                ht = feat(Xt[ti])
                logits = dom(GradRev.apply(torch.cat([hs, ht]), lam)).squeeze(1)
                lab = torch.cat([torch.zeros(len(idx), device=dev), torch.ones(len(idx), device=dev)])
                loss = loss + nn.functional.binary_cross_entropy_with_logits(logits, lab)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return reg(feat(Xt)).squeeze(1).cpu().numpy()


out = []
for held in sorted(d.group.unique()):
    src, te = d[d.group != held], d[d.group == held].copy()
    mu, sd = src[F].values.mean(0), src[F].values.std(0) + 1e-9
    Xs, Xt = (src[F].values - mu) / sd, (te[F].values - mu) / sd
    ym, ysd = src.Q.mean(), src.Q.std()
    ys = (src.Q.values - ym) / ysd
    for name, dann in (("MLP_src", False), ("DANN", True)):
        preds = np.mean([train(Xs, ys, Xt, dann) for _ in range(3)], axis=0)
        te[name] = preds * ysd + ym
    out.append(te)
    print("group", held, "done", flush=True)
t = pd.concat(out)
t[["cell", "group", "t", "Q", "M0", "MW_lin", "MW_quad", "MLP_src", "DANN"]].to_csv(OUT / "scored.csv.gz", index=False)
pols = ["M0", "MW_lin", "MW_quad", "MLP_src", "DANN"]
pg = pd.DataFrame({p: t.groupby("group").apply(lambda x: (x[p] - x.Q).abs().groupby(x.cell).mean().mean()) for p in pols})
res = dict(device=dev, mean_over_groups=pg.mean().round(5).to_dict(),
           DANN_beats_MW_quad=int((pg.DANN < pg.MW_quad).sum()), DANN_beats_MLP_src=int((pg.DANN < pg.MLP_src).sum()))
pg.round(4).to_csv(OUT / "per_group.csv")
json.dump(res, open(OUT / "summary.json", "w"), indent=2)
print(json.dumps(res, indent=1))
print(pg.round(4))
