"""Exact one-sided sign tests on the number of held-out groups where a method beats its comparator."""
import json
import sys
from math import comb
from pathlib import Path

import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
R = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=False)


def sign_p(wins, n):
    return sum(comb(n, i) for i in range(wins, n + 1)) / 2 ** n


def eqcell(path, cols):
    d = pd.read_csv(path)
    return pd.DataFrame({c: (d[c] - d.Q).abs().groupby([d.group, d.cell]).mean().groupby(level=0).mean() for c in cols})


rows = []


def add(name, a, b):
    rows.append(dict(comparison=name, wins=int((a < b).sum()), groups=int(len(a)), p=sign_p(int((a < b).sum()), len(a))))


for ds, f, w in (("UConn", "C11_UCONN_LOGO_2026-10-23_v1/scored_logo.csv.gz", "MW_quad"),
                 ("HUST masked", "C14_HUST_LOGO_2026-10-23_v1/scored_random.csv.gz", "MW_quad"),
                 ("KIT", "C31_KIT_LOGO_2026-10-24_usage/scored.csv.gz", "MW_quad"),
                 ("KIT with temperature", "C31_KIT_LOGO_2026-10-24_usage/scored.csv.gz", "MW_quadT")):
    g = eqcell(R / f, [w, "M0", "P1", "P2"])
    add(f"{ds}: window sensor vs training mean", g[w], g.M0)
    if ds != "KIT with temperature":
        add(f"{ds}: Eq. (2) vs reference + drift", g.P2, g.P1)
c91 = pd.read_csv(R / "C91_UCONN_CAUSAL_FUSION_2026-10-29_v1/per_group.csv")
c91 = c91[c91.variant == "causal_nested"]
c103 = pd.read_csv(R / "C103_UCONN_OWN_RATE_2026-10-31_v1/per_group.csv")
for T in (7, 14, 28, 56):
    x = c91[c91["T"] == T]
    add(f"UConn T={T}: bias-aware filter vs population-rate propagation", x.kf_bias.values, x.ref_drift.values)
    y = c103[c103["T"] == T]
    add(f"UConn T={T}: bias-aware filter vs own-rate propagation", y.kf_bias.values, y.own_ref.values)
c102 = pd.read_csv(R / "C102_UCONN_MEASURED_TEST_FUSION_2026-10-30_v1/per_group.csv")
x = c102[(c102.test == "measured") & (c102["T"] == 56)]
add("UConn measured test T=56: bias-aware filter vs propagation", x.kf_bias.values, x.ref_drift.values)
b = pd.read_csv(R / "C103b_KIT_HUST_OWN_RATE_2026-10-31_v1/per_group.csv")
for (ds, ke), x in b.groupby(["dataset", "k_every"]):
    add(f"{ds} k={ke}: bias-aware filter vs population-rate propagation", x.kf_bias.values, x.ref_drift.values)
    add(f"{ds} k={ke}: bias-aware filter vs own-rate propagation", x.kf_bias.values, x.own_ref.values)
    add(f"{ds} k={ke}: own-rate filter vs own-rate propagation", x.own_kf.values, x.own_ref.values)
t = pd.DataFrame(rows)
t.to_csv(OUT / "sign_tests.csv", index=False)
json.dump({r.comparison: dict(wins=f"{r.wins}/{r.groups}", p=round(r.p, 4)) for r in t.itertuples()},
          open(OUT / "summary.json", "w"), indent=2)
print(t.to_string(index=False))
