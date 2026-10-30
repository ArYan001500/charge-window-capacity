"""HUST: Kalman fusion with measured test cycles and the label-free fusion rule."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from c32_kit_kalman import job

C14, OUT = Path(sys.argv[1]), Path(sys.argv[2])


def prep(mode, k_test=50):
    d = pd.read_csv(C14 / f"scored_{mode}.csv.gz", usecols=["cell", "cyc", "group", "Q", "MW_quad"]).rename(columns={"MW_quad": "pred"})
    d = d.sort_values(["cell", "cyc"]).reset_index(drop=True)
    d["t"] = d.cyc.astype(float) * 86400
    first = d.groupby("cell").cyc.transform("min")
    is_test = (d.cyc - first) % k_test == 0
    labs = d[is_test][["cell", "t", "Q"]].rename(columns={"t": "t_lab"})
    d["near"] = ~is_test
    return d, {c: x.reset_index(drop=True) for c, x in labs.groupby("cell")}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=False)
    rows = []
    for mode in ("fixed", "random"):
        d, labs_by_cell = prep(mode)
        cells_by_group = {g: list(x.cell.unique()) for g, x in d.groupby("group")}
        s = d[d.near]
        passive = s.groupby("group").apply(lambda z: (z.pred - z.Q).abs().groupby(z.cell).mean().mean())
        with Pool(14) as pool:
            r = pool.map(job, [(g, k, d, labs_by_cell, cells_by_group) for k in (1, 2) for g in sorted(cells_by_group)])
        for x in r:
            x.update(mode=mode, passive=float(passive[x["group"]]))
        rows += r
    r = pd.DataFrame(rows)
    r["gain"] = r.passive - r.bias_kf
    out = {}
    for (mode, k), x in r.groupby(["mode", "k_every"]):
        x = x.copy()
        x["gain_src"] = [x[x.group != g].gain.mean() for g in x.group]
        x["rule"] = np.where(x.gain_src > 0, x.bias_kf, x.passive)
        best_setting = "bias_kf" if x.bias_kf.mean() < x.passive.mean() else "passive"
        rule_setting = "bias_kf" if (x.gain_src > 0).mean() > 0.5 else "passive"
        out[f"{mode}_k{k}"] = dict(passive=round(x.passive.mean(), 4), ref_drift=round(x.ref_drift.mean(), 4), bias_kf=round(x.bias_kf.mean(), 4),
                                   kf_beats_passive=f"{int((x.gain > 0).sum())}/{len(x)}", kf_beats_ref=f"{int((x.bias_kf < x.ref_drift).sum())}/{len(x)}",
                                   rule=round(x.rule.mean(), 4), rule_setting=rule_setting, best_setting=best_setting,
                                   R1_correct=rule_setting == best_setting,
                                   R2_groups_ok=f"{int((x.rule <= 1.1 * np.minimum(x.passive, x.bias_kf)).sum())}/{len(x)}")
        r.loc[x.index, "rule"] = x.rule
    out["R1_pass"] = f"{sum(v['R1_correct'] for v in out.values() if isinstance(v, dict))}/4"
    r.to_csv(OUT / "per_group.csv", index=False)
    json.dump(out, open(OUT / "summary.json", "w"), indent=2)
    print(json.dumps(out, indent=1))
