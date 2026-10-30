# Soft Sensing of Lithium-Ion Battery Capacity from 10-Minute Charging Windows

Code and results for the paper *Soft sensing of lithium-ion battery capacity from 10-minute charging windows under unseen usage*. A ridge model estimates the capacity of the next reference
test from 600 s of an ordinary charge, without a known starting state of charge or a fixed voltage window. A Kalman filter
with a per-cell bias state fuses these estimates with occasional reference tests, and a group-max interval rule is
calibrated on the worst training usage group. All models are tested on whole usage groups that were not used for
training.

Aryan Ashoori · [ORCID](https://orcid.org/0009-0001-2226-3894) · aryanashoori0@gmail.com

Citation: details will be added when the paper is published (see `CITATION.cff`).

## Quick start

Python 3.12 or newer, Windows or Linux. No GPU is needed.

```console
python -m pip install -r requirements.txt
python -m chargewin reproduce --tier paper
python -m chargewin reproduce --tier quick
```

| Tier | What it does | Run time | Download |
|---|---|---|---|
| `paper` | Checks 117 numbers printed in the paper against the result files in `paper/` and redraws Figures 1 to 7 | about 1 min | none |
| `quick` | Downloads 14 HUST cells, runs the window sensor (C14) and the HUST filter (C54), compares with `expected/quick.json` | about 10 min | 0.22 GB |
| `main` | All results of Sections 6.1 to 6.4 and Table A.3 from the raw data; each result folder is compared with `paper/` | 3 to 5 h (estimate) | 16.3 GB |
| `full` | `main` plus the neural-network comparisons, the studies of Appendix C and the CALCE illustration | 6 to 10 h (estimate) | 16.5 GB |

The `main` and `full` tiers also need PyTorch (small perceptrons in C86 and C87, networks in the `full` tier). The CPU
build is enough and avoids several GB of GPU libraries:

```console
python -m pip install -r requirements-nn.txt --index-url https://download.pytorch.org/whl/cpu
python -m chargewin reproduce --tier main --data-dir ./data
python -m chargewin reproduce --tier full --data-dir ./data
pytest -q
```

Every tier ends with PASS or FAIL. Runs are written to `runs/<tier>/` (change with `--out-dir`); finished steps are
skipped when a run is restarted, and each step keeps its log in `runs/<tier>/logs/`. Results of deterministic steps
must agree with `paper/` to 1e-6. Steps that train small neural networks (PyTorch) can differ between CPU and GPU, so
only their `summary.json` is compared, with a tolerance of 2e-3. Three nonlinear least-squares fits (C49 pulse features, C74 and C75 two-electrode models) depend on the SciPy version: with the pinned SciPy 1.17.1, C74 and C75 reproduce `paper/` exactly, whereas the poorly identified circuit parameters of C49 differ (C49 was run with SciPy 1.18.1, with which it is exact). C49 is therefore not compared; its use in C49b is compared on `summary.json` with a tolerance of 1e-2 (largest change 0.006 percentage points), and the numbers printed in the paper do not change.

## Data

All four datasets are public. HUST and CALCE are downloaded automatically. UConn and KIT cannot be fetched by a script
(the UConn server refuses non-browser downloads; the KIT repository asks to accept its terms of use first), so please
download these files in a browser and place them as shown. Every file is checked against the size and MD5 listed in
`chargewin/data_manifest.json`.

| Dataset | Files | Where to put them | Source and license |
|---|---|---|---|
| UConn NMC/graphite, 44 cells | `rpt_data.zip`, `cycling_data.zip`, `cycling_parameters.csv` | `data/uconn/` | [UConn digital commons](https://digitalcommons.lib.uconn.edu/reil_datasets/2/), CC BY 4.0 |
| HUST LFP, 77 cells | `our_data.zip` (automatic) | `data/hust/` | [10.17632/nsc7hnsg4s.2](https://doi.org/10.17632/nsc7hnsg4s.2), CC BY 4.0 |
| KIT NMC/C-SiO, result data | `10.35097-1969.tar` (button "Download" of the dataset) | `data/kit/` | [10.35097/1969](https://doi.org/10.35097/1969), CC BY 4.0 |
| KIT NMC/C-SiO, log data | `cell_log_age_ultracompr.7z` (single file of the dataset) | `data/kit/` | [10.35097/kww7jv8ajuvchcah](https://doi.org/10.35097/kww7jv8ajuvchcah), CC BY 4.0 |
| CALCE partial cycling | `SOC_20-80_2C.zip`, `SOC_20-80_HalfC.zip` (automatic) | `data/calce/` | [CALCE battery data](https://calce.umd.edu/battery-data), no license stated |

The KIT logs are unpacked by the runner (7-Zip or `py7zr`; the 30-s files only, 240 cells). The raw CALCE files store
MATLAB tables, so the CALCE illustration (Fig. 2a, b) needs MATLAB for one export step (`matlab/export_calce_numeric.m`);
without MATLAB the `full` tier skips it. The `main` tier redraws Figures 1, 3, 4, 6 and 7; Figures 2 and 5 need results of the `full` tier. `paper/` holds the result files behind every figure, including the held-out
predictions, and the paper tier draws all figures from them.

## How the code maps to the paper

The paper refers to each analysis by its protocol number (Tables 3, A.1 to A.3, B.1 and C.1). The script of protocol
`Cxx` is `chargewin/steps/cxx_*.py`, and its results are in `paper/results/Cxx_*`. `chargewin/pipeline.py` lists every
step with its inputs and outputs in run order, and `chargewin/numbers.py` links each checked number of the paper to the
result file it comes from.

| Paper | Protocols |
|---|---|
| Window sensor and Eq. (2), Section 6.1, Table 3 | C11 (UConn), C14 (HUST), C31 (KIT) |
| Integral and curvature descriptors | C61, C61b, C62 |
| Adapted competitors, Section 6.2, Table B.1 | C78, C78b, C79, C87, C55, C85, C98 |
| Fusion with sparse tests, Section 6.3 | C91 (UConn), C93 (KIT, HUST), C54, C95, C102, C103 (own-rate baseline) |
| Intervals under shift, Section 6.4 | C90, C92, C99, C101 |
| Robustness analyses, Table A.3 | C96 to C104 |
| Negative results, Appendix C | C57, C60, C65, C67 to C70, C72, C74, C75, C80 |

Earlier implementations that the paper reports as superseded (C24, C25, C32, C56) are kept and run in the `full` tier.
Two superseded result folders that were replaced after a failed check are kept in `paper/superseded/`.

Two inputs of the `full` tier come from external tools and are copied from `paper/`: the forecasts of three pretrained
time-series models used by the trigger study C69 (`c69_fm_forecasts.py` regenerates them given the model checkpoints)
and the half-cell curves of C74 (`c74_export_ocp.py`, needs PyBaMM).

## Contents

- `chargewin/steps/`: one script per protocol; each reads the run folder (`data/`, `tables/`, `results/`).
- `chargewin/figures/`: the figure scripts.
- `chargewin/`: the runner, data download and checks, and the number check.
- `paper/`: the result files of the paper and the 791-byte UConn group table used by Figure 2 (no other raw data).
- `expected/`: reference values of the quick tier.
- `matlab/`: numeric export of the raw CALCE files.
- `tests/`: checks of the step list, the data manifest and the paper numbers.

## License

Code: MIT (see `LICENSE`). Data: from the sources above under their own terms; apart from the UConn group table (CC BY 4.0), no raw data are included here.
