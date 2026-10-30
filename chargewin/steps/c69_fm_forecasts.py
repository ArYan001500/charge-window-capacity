"""Forecasts from pretrained time-series models used by the trigger study."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

INP, MODELS, OUT, which = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
OUT.mkdir(parents=True, exist_ok=True)
H = 120
W = pd.read_csv(INP / "window_preds.csv")
L = pd.read_csv(INP / "checkup_days.csv")

contexts, keys = [], []
for c, x in W.groupby("cell"):
    x = x.sort_values("day")
    d0 = np.floor(x.day.min())
    s = x.assign(dd=np.floor(x.day - d0).astype(int)).groupby("dd").pred.median()
    grid = np.arange(0, s.index.max() + 1)
    full = np.interp(grid, s.index.values, s.values)
    for kd in L[L.cell == c].day.values:
        n = int(np.floor(kd - d0))
        if n < 7:
            continue
        contexts.append(full[: min(n, len(full))].astype(np.float32))
        keys.append((c, float(kd), d0))
print(which, "contexts", len(contexts), flush=True)
t0 = time.time()
dev = "cuda" if torch.cuda.is_available() else "cpu"
if which == "chronos2":
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained(str(MODELS / "chronos2"), device_map=dev)
    meds = []
    for i in range(0, len(contexts), 256):
        q, _ = pipe.predict_quantiles([torch.tensor(c) for c in contexts[i:i + 256]], prediction_length=H, quantile_levels=[0.5])
        meds += [np.asarray(z.squeeze().float().cpu().numpy()).reshape(-1)[:H] for z in q]
elif which == "tirex":
    import tirex
    from tirex.base import PretrainedModel
    model = PretrainedModel.REGISTRY["TiRex"].from_pretrained(str(MODELS / "tirex2" / "model.ckpt"), backend="torch", device=dev)
    meds = []
    for i in range(0, len(contexts), 256):
        q, m = model.forecast([torch.tensor(c) for c in contexts[i:i + 256]], prediction_length=H, output_type="numpy")
        meds += [np.asarray(r).reshape(-1)[:H] for r in m]
elif which == "moirai2":
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    mod = Moirai2Module.from_pretrained(str(MODELS / "moirai2"))
    meds = []
    for c in contexts:
        f = Moirai2Forecast(prediction_length=H, target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
                            context_length=len(c), module=mod)
        out = f.predict(past_target=[c[:, None]])
        arr = np.asarray(out)
        meds.append(np.median(arr.reshape(arr.shape[0], -1, H) if arr.ndim > 2 else arr, axis=tuple(range(arr.ndim - 1))).reshape(-1)[:H]
                    if arr.ndim > 1 else arr.reshape(-1)[:H])
else:
    raise ValueError(which)
rows = [dict(cell=k[0], kday=k[1], d0=k[2], h=h + 1, med=float(m[h])) for k, m in zip(keys, meds) for h in range(len(m))]
pd.DataFrame(rows).to_csv(OUT / f"fm_{which}.csv.gz", index=False)
print(which, "done", round(time.time() - t0, 1), "s", flush=True)
