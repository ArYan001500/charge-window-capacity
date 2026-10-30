"""Shao network restricted to earlier windows."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT, OUT, NAME, ARCH, SEED = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)


class ShaoNet(nn.Module):
    def __init__(self, h=32, bidir=True):
        super().__init__()
        self.bidir, self.h = bidir, h
        self.lstm = nn.LSTM(1, h, num_layers=2, batch_first=True, bidirectional=bidir)
        self.W1 = nn.Linear(h, h)
        if bidir:
            self.W2 = nn.Linear(h, h)
        self.out = nn.Linear(2 * h if bidir else h, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        q1 = o[..., :self.h]
        o1 = q1 + torch.sigmoid(self.W1(torch.tanh(q1)))
        if not self.bidir:
            return self.out(o1).squeeze(-1)
        q2 = o[..., self.h:]
        o2 = q2 + torch.sigmoid(self.W2(torch.tanh(q2)))
        return self.out(torch.cat([o1, o2], -1)).squeeze(-1)


def windows(name):
    if name == "UConn":
        s = pd.read_csv(ROOT / "results/C11_UCONN_LOGO_2026-10-23_v1/scored_logo.csv.gz", usecols=["cell", "group", "t", "Q", "MW_quad"]).rename(columns={"MW_quad": "ours"})
        cyc = pd.read_csv(ROOT / "tables/C10_UCONN_2026-10-23_v1/uconn_cycling_table.csv.gz").rename(columns={"date": "t"})
        d = s.merge(cyc, on=["cell", "t"], how="inner")
        d["tt"] = pd.to_datetime(d.t).astype("int64") / 8.64e13
        step = 10
    else:
        s = pd.read_csv(ROOT / "results/C31_KIT_LOGO_2026-10-24_usage/scored.csv.gz", usecols=["cell", "group", "t", "Q", "MW_quadT"]).rename(columns={"MW_quadT": "ours"})
        d = s.merge(pd.read_csv(ROOT / "tables/C31_KIT_2026-10-24_v1/kit_windows.csv.gz"), on=["cell", "t"], how="inner")
        d["tt"] = d.t / 86400
        step = 30
    ts = list(range(0, 601, step))
    P = d[[f"V{t}" for t in ts]].values * np.abs(d[[f"I{t}" for t in ts]].values)
    d["E"] = ((P[:, 1:] + P[:, :-1]) / 2 * step / 3600).sum(1)
    return d.dropna(subset=["E", "ours"]).sort_values(["cell", "tt"]).reset_index(drop=True)


def chunks(d, L=32):
    out = []
    for c, x in d.groupby("cell"):
        idx = x.index.values
        for i in range(0, len(idx), L):
            out.append(idx[i:i + L])
    return out


t0 = time.time()
d = windows(NAME)
E, Q = d.E.values, d.Q.values
pred = np.full(len(d), np.nan)
for g in sorted(d.group.unique()):
    src, tgt = d[d.group != g], d[d.group == g]
    em, es, qm, qsd = src.E.mean(), src.E.std(), src.Q.mean(), src.Q.std()
    net = ShaoNet(bidir=(ARCH == "bi")).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    ch = chunks(src)
    for ep in range(300):
        np.random.shuffle(ch)
        for i in range(0, len(ch), 64):
            batch = ch[i:i + 64]
            Lm = max(len(b) for b in batch)
            xb = np.zeros((len(batch), Lm, 1), np.float32)
            yb = np.zeros((len(batch), Lm), np.float32)
            mb = np.zeros((len(batch), Lm), np.float32)
            for k, b in enumerate(batch):
                xb[k, :len(b), 0] = (E[b] - em) / es
                yb[k, :len(b)] = (Q[b] - qm) / qsd
                mb[k, :len(b)] = 1
            xb, yb, mb = (torch.tensor(a, device=dev) for a in (xb, yb, mb))
            opt.zero_grad()
            loss = (((net(xb) - yb) ** 2) * mb).sum() / mb.sum()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        for b in chunks(tgt):
            xb = torch.tensor(((E[b] - em) / es)[None, :, None], dtype=torch.float32, device=dev)
            pred[b] = net(xb).squeeze(0).cpu().numpy() * qsd + qm
    print(NAME, ARCH, SEED, "group", g, round(time.time() - t0), "s", flush=True)
d["shao"] = pred
d[["cell", "group", "t", "Q", "ours", "shao"]].to_csv(OUT / f"pred_{NAME}_{ARCH}_s{SEED}.csv.gz", index=False)
json.dump(dict(dataset=NAME, arch=ARCH, seed=SEED, device=dev, n=len(d), seconds=round(time.time() - t0)),
          open(OUT / f"run_{NAME}_{ARCH}_s{SEED}.json", "w"))
