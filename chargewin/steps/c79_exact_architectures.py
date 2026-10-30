"""Rebuilt Guo and Shao network architectures on the same windows and segments."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8")
ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=False)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(79)
np.random.seed(79)
sys.path.insert(0, str(Path(__file__).parent))
NBIN, QS = 12, 2.5
LABELS = np.round(np.arange(3.00, 4.2001, 0.01), 2)


def cellmae(df, col):
    return df.assign(ae=(df[col] - df.Q).abs()).groupby(["group", "cell"]).ae.mean().reset_index()


def eqmae(df, col):
    return float(cellmae(df, col).groupby("group").ae.mean().mean())


def ci_diff(df, a, b, seed=79):
    t = cellmae(df, a).rename(columns={"ae": "a"}).merge(cellmae(df, b).rename(columns={"ae": "b"}), on=["group", "cell"])
    r = np.random.default_rng(seed)
    diffs = [np.mean([(lambda z: (z.a - z.b).mean())(x.sample(len(x), replace=True, random_state=int(r.integers(1e9)))) for _, x in t.groupby("group")])
             for _ in range(2000)]
    return [round(float(v), 4) for v in np.percentile(diffs, [2.5, 97.5])]


class GuoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(2, 15, batch_first=True)
        self.fc1 = nn.Linear(15, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        h, _ = self.gru(x)
        f = torch.relu(self.fc1(h))
        return torch.sigmoid(self.fc2(f)).squeeze(-1), f.mean(1)


def mmd(a, b, bw=10.0):
    def k(x, y):
        return torch.exp(-torch.cdist(x, y) ** 2 / (2 * bw ** 2))
    return k(a, a).mean() + k(b, b).mean() - 2 * k(a, b).mean()


def guo_train_levels(curves_by_level):
    """curves_by_level: list (ordered by capacity) of arrays [n, len(LABELS)] of q (Ah, NaN where not observed)."""
    xin = torch.tensor(np.c_[(LABELS - 3.0) / 1.2, np.full(len(LABELS), 0.75 / QS)], dtype=torch.float32, device=dev)
    models, prev = [], None
    for lvl, Y in enumerate(curves_by_level):
        net = GuoNet().to(dev)
        if prev is not None:
            net.load_state_dict(prev.state_dict())
        opt = torch.optim.SGD(net.parameters(), lr=0.03, momentum=0.9)
        y = torch.tensor(np.nan_to_num(Y / QS, nan=0.0), dtype=torch.float32, device=dev)
        m = torch.tensor(~np.isnan(Y), dtype=torch.float32, device=dev)
        X = xin.unsqueeze(0).expand(len(Y), -1, -1)
        if prev is not None:
            Ys = curves_by_level[lvl - 1]
            Xs = xin.unsqueeze(0).expand(len(Ys), -1, -1)
        for _ in range(400):
            opt.zero_grad()
            p, ft = net(X)
            loss = ((p - y) ** 2 * m).sum() / m.sum()
            if prev is not None:
                _, fs = net(Xs)
                d = mmd(fs, ft)
                loss = loss + 5.0 * d / torch.tanh(torch.clamp(0.5 - d, min=1e-3))
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            curve = net(xin.unsqueeze(0))[0].squeeze(0).cpu().numpy() * QS
        models.append(np.maximum.accumulate(curve))
        prev = net
    return models


def run_guo():
    sys.argv = ["c61", str(ROOT), str(OUT / "_unused")]
    src61 = open(Path(__file__).with_name("c61_position_integral_features.py"), encoding="utf-8").read().split("res = {}")[0]
    ns = {}
    exec(src61.replace("OUT.mkdir(parents=True, exist_ok=False)", ""), ns)
    c, QC, V, segment, feats, ridge_fit, pick_alpha = ns["c"], ns["QC"], ns["V"], ns["segment"], ns["feats"], ns["ridge_fit"], ns["pick_alpha"]
    Q_on_labels = []
    for _, r in c.iterrows():
        qv = r[QC].values.astype(float)
        ok = ~np.isnan(qv)
        q = np.maximum.accumulate(qv[ok])
        q = q - q[0]
        vv = V[ok]
        lab = np.interp(LABELS, vv, q, left=np.nan, right=np.nan)
        Q_on_labels.append(lab)
    Q_on_labels = np.array(Q_on_labels)
    out = {}
    for pos in ("LOW", "MID", "HIGH"):
        rows = []
        for i, r in c.iterrows():
            s = segment(r[QC].values.astype(float), pos)
            if s is None:
                continue
            f0, f1 = feats(s)
            rows.append(dict(idx=i, cell=r.cell, group=r.group, Q=r.Q_ref, v1=s[0], v2=s[-1], f0=f0, f1=f1))
        x = pd.DataFrame(rows)
        X = np.hstack([np.array(x.f0.tolist()), np.array(x.f1.tolist())[:, :-1]])
        x["ours"], x["guo_exact"] = np.nan, np.nan
        for g in sorted(x.group.unique()):
            tr, te = (x.group != g).values, (x.group == g).values
            x.loc[te, "ours"] = ridge_fit(X[tr], x.Q.values[tr], pick_alpha(X[tr], x.Q.values[tr], x.group.values[tr]))(X[te])
            if pos == "LOW":
                srcidx = c.index[c.group != g]
                qs = c.loc[srcidx, "Q_ref"].values
                edges = np.quantile(qs, np.linspace(0, 1, NBIN + 1))
                lv_q, lv_curves = [], []
                for b in range(NBIN):
                    m = (qs >= edges[b]) & (qs <= edges[b + 1])
                    lv_q.append(qs[m].mean())
                    lv_curves.append(Q_on_labels[srcidx[m]])
                GUO_CACHE[g] = (np.array(lv_q), guo_train_levels(lv_curves))
            lv_q, curves = GUO_CACHE[g]
            for j in x.index[te]:
                dq_b = np.array([np.interp(x.at[j, "v2"], LABELS, cv) - np.interp(x.at[j, "v1"], LABELS, cv) for cv in curves])
                order = np.argsort(dq_b)
                x.at[j, "guo_exact"] = float(np.interp(0.375, dq_b[order], lv_q[order]))
            print("guo", pos, "group", g, flush=True)
        out[pos] = dict(ours_F0F1=round(eqmae(x, "ours"), 4), guo_exact=round(eqmae(x, "guo_exact"), 4), ours_minus_guo_CI95=ci_diff(x, "ours", "guo_exact"), n=len(x))
    return out


GUO_CACHE = {}


class ShaoNet(nn.Module):
    def __init__(self, h=32):
        super().__init__()
        self.lstm = nn.LSTM(1, h, num_layers=2, batch_first=True, bidirectional=True)
        self.h = h
        self.W1, self.W2 = nn.Linear(h, h), nn.Linear(h, h)
        self.out = nn.Linear(2 * h, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        q1, q2 = o[..., :self.h], o[..., self.h:]
        o1 = q1 + torch.sigmoid(self.W1(torch.tanh(q1)))
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


def run_shao(name):
    d = windows(name)
    d["shao_exact"] = np.nan
    for g in sorted(d.group.unique()):
        src, tgt = d[d.group != g], d[d.group == g]
        em, es, qm, qsd = src.E.mean(), src.E.std(), src.Q.mean(), src.Q.std()
        net = ShaoNet().to(dev)
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
                    xb[k, :len(b), 0] = (d.loc[b, "E"].values - em) / es
                    yb[k, :len(b)] = (d.loc[b, "Q"].values - qm) / qsd
                    mb[k, :len(b)] = 1
                xb, yb, mb = (torch.tensor(a, device=dev) for a in (xb, yb, mb))
                opt.zero_grad()
                loss = (((net(xb) - yb) ** 2) * mb).sum() / mb.sum()
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            for b in chunks(tgt):
                xb = torch.tensor(((d.loc[b, "E"].values - em) / es)[None, :, None], dtype=torch.float32, device=dev)
                d.loc[b, "shao_exact"] = net(xb).squeeze(0).cpu().numpy() * qsd + qm
        print("shao", name, "group", g, flush=True)
    return dict(ours=round(eqmae(d, "ours"), 4), shao_exact=round(eqmae(d, "shao_exact"), 4), ours_minus_shao_CI95=ci_diff(d, "ours", "shao_exact"), n=len(d))


if __name__ == "__main__":
    res = {"device": dev, "guo_exact_slow_segments": run_guo()}
    for n in ("UConn", "KIT"):
        res[f"shao_exact_{n}"] = run_shao(n)
    res["X1_pass_LOW"] = res["guo_exact_slow_segments"]["LOW"]["ours_minus_guo_CI95"][1] < 0
    res["X2_pass"] = all(res[f"shao_exact_{n}"]["ours_minus_shao_CI95"][1] < 0 for n in ("UConn", "KIT"))
    json.dump(res, open(OUT / "summary.json", "w"), indent=2)
    print(json.dumps(res, indent=1))
