"""UConn: pulse-test table with the perceptron window estimate in place of the ridge estimate."""
import sys
from pathlib import Path

import pandas as pd

C19, C15, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
a = pd.read_csv(C19 / "scored.csv.gz", parse_dates=["t"])
m = pd.read_csv(C15 / "scored.csv.gz", parse_dates=["t"])[["cell", "t", "MLP_src"]]
x = a.merge(m, on=["cell", "t"], how="inner")
print(len(a), len(x))
x["W_ridge"] = x.W
x["W"] = x.MLP_src
x.to_csv(OUT / "scored.csv.gz", index=False)
