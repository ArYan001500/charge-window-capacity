"""KIT: tuned domain-adversarial network against a tuned source-only perceptron."""
import json
import sys
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

W_DIR, OUT = Path(sys.argv[1]), Path(sys.argv[2])
DANN_GRID = [("DANN", lm, ep, wd) for lm, ep, wd in product((0.03, 0.1, 0.3, 1.0), (60, 150), (64, 128))]
MLP_GRID = [("MLP", 0.0, ep, wd) for ep, wd in product((60, 150), (64, 128))]
SEEDS = (0, 1)


def load():
    w = pd.read_csv(W_DIR / "kit_windows.csv.gz").sort_values("t")
    lab = pd.read_csv(W_DIR / "kit_checkup_labels.csv").sort_values("t_lab")
    d = pd.merge_asof(w, lab.rename(columns={"t_lab": "t_next"}), left_on="t", right_on="t_next", by="cell", direction="forward")
    d = d[(d.t_next - d.t) <= 7 * 86400].dropna(subset=["Q"]).copy()
    d = d.sample(frac=1.0, random_state=1).groupby(["cell", "t_next"]).head(10).copy()
    combo = d.apply(lambda r: "WLTP" if r.age_type == 3 else f"{r.vmax}_{r.vmin}_{r.chg}_{r.dis}", axis=1)
    d["group"] = combo.astype("category").cat.codes
    d["dVa"], d["dVb"] = d.V60 - d.V0, d.V600 - d.V300
    d["Tmean"] = d[[f"T{s}" for s in range(0, 601, 30)]].mean(1)
    return d.reset_index(drop=True)


F = [f"V{s}" for s in range(0, 601, 60)] + ["Q600", "dVa", "dVb", "I0", "I300", "I600", "Tmean"]


class GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def train(Xs, ys, Xt, kind, lam_max, epochs, width, seed, bs=256):
    torch.manual_seed(seed)
    dann = kind == "DANN"
    feat = nn.Sequential(nn.Linear(Xs.shape[1], width), nn.ReLU(), nn.Linear(width, width // 2), nn.ReLU())
    reg = nn.Linear(width // 2, 1)
    dom = nn.Sequential(nn.Linear(width // 2, 32), nn.ReLU(), nn.Linear(32, 1))
    params = list(feat.parameters()) + list(reg.parameters()) + (list(dom.parameters()) if dann else [])
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
    Xs, ys, Xt = (torch.tensor(a, dtype=torch.float32) for a in (Xs, ys, Xt))
    steps = max(1, len(Xs) // bs)
    for ep in range(epochs):
        lam = lam_max * (2 / (1 + np.exp(-10 * ep / epochs)) - 1)
        perm = torch.randperm(len(Xs))
        for s in range(steps):
            idx = perm[s * bs:(s + 1) * bs]
            hs = feat(Xs[idx])
            loss = ((reg(hs).squeeze(1) - ys[idx]) ** 2).mean()
            if dann:
                ht = feat(Xt[torch.randint(0, len(Xt), (len(idx),))])
                logits = dom(GradRev.apply(torch.cat([hs, ht]), lam)).squeeze(1)
                labd = torch.cat([torch.zeros(len(idx)), torch.ones(len(idx))])
                loss = loss + nn.functional.binary_cross_entropy_with_logits(logits, labd)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return reg(feat(Xt)).squeeze(1).numpy()


def eqmae(x, p):
    return float((pd.Series(p, index=x.index) - x.Q).abs().groupby(x.cell).mean().mean())


def task(args):
    """One (outer group, role, pseudo-target or None, config): returns MAE and (for outer) predictions."""
    torch.set_num_threads(1)
    g, pt, cfg = args
    src = D[(D.group != g) & (D.group != pt)] if pt is not None else D[D.group != g]
    te = D[D.group == (pt if pt is not None else g)]
    mu, sd = src[F].values.mean(0), src[F].values.std(0) + 1e-9
    ym, ysd = src.Q.mean(), src.Q.std()
    Xs, Xt = (src[F].values - mu) / sd, (te[F].values - mu) / sd
    p = np.mean([train(Xs, (src.Q.values - ym) / ysd, Xt, *cfg, seed=s) for s in SEEDS], axis=0) * ysd + ym
    return dict(g=g, pt=pt, cfg=cfg, mae=eqmae(te, p), pred=p if pt is None else None, idx=te.index.values if pt is None else None)


D = load()

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(58)
    groups = sorted(D.group.unique())
    pts = {g: list(rng.choice([h for h in groups if h != g], 2, replace=False)) for g in groups}
    jobs = [(g, pt, cfg) for g in groups for cfg in DANN_GRID + MLP_GRID for pt in pts[g] + [None]]
    print("jobs", len(jobs), flush=True)
    with Pool(18) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(task, jobs, chunksize=1)):
            res.append(r)
            if i % 50 == 0:
                print(i, flush=True)
    R = pd.DataFrame([{k: v for k, v in r.items() if k not in ("pred", "idx")} for r in res])
    R["cfg"] = R.cfg.astype(str)
    R.to_csv(OUT / "all_runs.csv", index=False)
    preds = {(r["g"], str(r["cfg"])): (r["idx"], r["pred"]) for r in res if r["pt"] is None}
    rows, cellrows = [], []
    for g in groups:
        inner = R[R.pt.notna() & (R.g == g)].groupby("cfg").mae.mean()
        outer = R[R.pt.isna() & (R.g == g)].set_index("cfg").mae
        for kind, grid in (("DANN", DANN_GRID), ("MLP", MLP_GRID)):
            names = [str(c) for c in grid]
            sel = inner[names].idxmin()
            orc = outer[names].idxmin()
            rows.append(dict(group=g, kind=kind, selected=sel, sel_mae=outer[sel], oracle=orc, oracle_mae=outer[orc]))
            idx, p = preds[(g, sel)]
            x = D.loc[idx]
            cellrows.append(pd.DataFrame({"group": g, "kind": kind, "cell": x.cell.values, "ae": np.abs(p - x.Q.values)}))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "per_group.csv", index=False)
    C = pd.concat(cellrows).groupby(["group", "kind", "cell"]).ae.mean().unstack("kind").reset_index()
    diffs, brng = [], np.random.default_rng(580)
    for _ in range(2000):
        diffs.append(np.mean([(lambda s: (s.DANN - s.MLP).mean())(x.sample(len(x), replace=True, random_state=int(brng.integers(1e9))))
                              for _, x in C.groupby("group")]))
    P = T.pivot(index="group", columns="kind", values=["sel_mae", "oracle_mae"])
    out = dict(selected_DANN=round(float(P.sel_mae.DANN.mean()), 4), selected_MLP=round(float(P.sel_mae.MLP.mean()), 4),
               oracle_DANN=round(float(P.oracle_mae.DANN.mean()), 4), oracle_MLP=round(float(P.oracle_mae.MLP.mean()), 4),
               T1_DANN_minus_MLP_CI95=[round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
               T1_DANN_wins=f"{int((P.sel_mae.DANN < P.sel_mae.MLP).sum())}/{len(P)}",
               T1_pass=bool(P.sel_mae.DANN.mean() < P.sel_mae.MLP.mean()),
               T2_oracleDANN_wins_vs_selMLP=f"{int((P.oracle_mae.DANN < P.sel_mae.MLP).sum())}/{len(P)}",
               T2_oracleDANN_over_selMLP=round(float(P.oracle_mae.DANN.mean() / P.sel_mae.MLP.mean()), 3),
               C33_reference=dict(MLP_src=0.0665, DANN=0.0808, MW_quadT=0.1028),
               selected_configs=T.groupby("kind").selected.agg(lambda s: s.value_counts().to_dict()).to_dict())
    json.dump(out, open(OUT / "summary.json", "w"), indent=2)
    print(json.dumps(out, indent=1))
