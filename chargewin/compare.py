"""Numerical comparison of a result folder with the shipped one."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _flat(v, f"{p}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _flat(v, f"{p}[{i}]")
    else:
        yield p, o


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def compare(reference, actual, names=None):
    """Largest absolute difference over all CSV and JSON files of `reference`, and a list of problems."""
    reference, actual = Path(reference), Path(actual)
    worst, problems = 0.0, []
    files = [reference / n for n in names] if names else sorted(reference.rglob("*"))
    for f in files:
        if f.is_dir() or not f.name.endswith((".csv", ".csv.gz", ".json")):
            continue
        g = actual / f.relative_to(reference)
        if not g.exists():
            problems.append(f"missing {g.relative_to(actual)}")
            continue
        if f.name.endswith(".json"):
            a = dict(_flat(json.loads(f.read_text(encoding="utf-8"))))
            b = dict(_flat(json.loads(g.read_text(encoding="utf-8"))))
            for k in sorted(set(a) | set(b)):
                if k.endswith("/seconds"):
                    continue
                x, y = a.get(k), b.get(k)
                if _num(x) and _num(y):
                    d = 0.0 if (np.isnan(x) and np.isnan(y)) else abs(float(x) - float(y))
                    worst = max(worst, d)
                elif x != y:
                    problems.append(f"{f.name}{k}: {x!r} vs {y!r}"[:160])
        else:
            a, b = pd.read_csv(f, low_memory=False), pd.read_csv(g, low_memory=False)
            if a.shape != b.shape or list(a.columns) != list(b.columns):
                problems.append(f"{f.name}: shape {a.shape} vs {b.shape}")
                continue
            if not a.astype(str).equals(b.astype(str)):
                a = a.sort_values(list(a.columns), ignore_index=True)
                b = b.sort_values(list(b.columns), ignore_index=True)
            for c in a.columns:
                if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
                    x, y = a[c].to_numpy(float), b[c].to_numpy(float)
                    if not np.array_equal(np.isnan(x), np.isnan(y)):
                        problems.append(f"{f.name}:{c}: missing values differ")
                        continue
                    ok = ~np.isnan(x)
                    if ok.any():
                        worst = max(worst, float(np.abs(x[ok] - y[ok]).max()))
                elif not a[c].astype(str).equals(b[c].astype(str)):
                    problems.append(f"{f.name}:{c}: text differs")
    return worst, problems
