"""Half-cell open-circuit potential curves exported from PyBaMM parameter sets."""
import sys
import numpy as np
import pandas as pd
import pybamm

OUT = sys.argv[1]
sto = np.linspace(0.001, 0.999, 500)
rows = []
for ps in ("Chen2020", "Mohtat2020", "OKane2022", "Prada2013", "Ecker2015", "Marquis2019"):
    try:
        pv = pybamm.ParameterValues(ps)
    except Exception as e:
        print(ps, "FAIL", e)
        continue
    for el in ("Positive", "Negative"):
        f = pv[f"{el} electrode OCP [V]"]
        vals = []
        for x in sto:
            try:
                if callable(f):
                    try:
                        v = f(pybamm.Scalar(x))
                    except TypeError:
                        v = f(pybamm.Scalar(x), pybamm.Scalar(298.15))
                    v = pv.process_symbol(v).evaluate() if isinstance(v, pybamm.Symbol) else v
                else:
                    v = f
                vals.append(float(np.squeeze(v)))
            except Exception as e:
                vals.append(np.nan)
        rows += [dict(set=ps, electrode=el, sto=s, U=u) for s, u in zip(sto, vals)]
        print(ps, el, "nan share", np.mean(np.isnan(vals)), "range", np.nanmin(vals), np.nanmax(vals))
pd.DataFrame(rows).to_csv(f"{OUT}/halfcell_ocp.csv", index=False)
