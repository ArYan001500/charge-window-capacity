"""Reproduction tiers."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from chargewin.compare import compare
from chargewin.pipeline import STEPS

PKG = Path(__file__).resolve().parent
REPO = PKG.parent
PAPER = REPO / "paper"
TOL = {"main": 1e-6, "full": 1e-6}
NN_TOL = 2e-3
QUICK_STEPS = ["C14", "C54"]
# inputs of the full tier made with external tools: forecasts of three pretrained time-series models
# (c69_fm_forecasts.py) and PyBaMM half-cell curves (c74_export_ocp.py)
SEEDED = ["results/C69_FM_FORECASTS_2026-10-24_v1", "tables/C74_HALFCELL_OCP"]


def _link(target, link):
    """Directory link from the run folder to the data folder (junction on Windows)."""
    if link.exists() or link.is_symlink():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    else:
        os.symlink(target, link, target_is_directory=True)


def _env():
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8")
    return env


def run_step(s, root, log):
    done = root / s["out"] / f".done_{s['id']}"
    if done.exists():
        print(f"skip {s['id']} (finished earlier)", flush=True)
        return
    print(f"run  {s['id']}: {s['script']}", flush=True)
    t0 = time.monotonic()
    with open(log / f"{s['id']}.log", "w", encoding="utf-8") as f:
        p = subprocess.run([sys.executable, "-B", str(PKG / "steps" / s["script"]), *s["args"]], cwd=root, env=_env(),
                           stdout=f, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f"step {s['id']} failed; see {log / (s['id'] + '.log')}")
    (root / s["out"]).mkdir(parents=True, exist_ok=True)
    done.write_text(f"{time.monotonic() - t0:.0f} s\n")
    print(f"     {s['id']} finished in {time.monotonic() - t0:.0f} s", flush=True)


def check_against_paper(steps, root, tol):
    rows, failed, seen = [], [], set()
    for s in steps:
        if s["out"] in seen or not s["out"].startswith("results/"):
            continue
        seen.add(s["out"])
        ref = PAPER / s["out"]
        if not ref.exists():
            continue
        nn = "import torch" in (PKG / "steps" / s["script"]).read_text(encoding="utf-8")
        names = [p.relative_to(ref) for p in ref.rglob("summary.json")] if nn else None
        worst, problems = compare(ref, root / s["out"], names)
        ok = worst <= (NN_TOL if nn else tol[s["tier"]]) and not problems
        rows.append((s["out"].split("/", 1)[1], worst, "PASS" if ok else "FAIL"))
        if not ok:
            failed.append((s["out"], problems[:5]))
    print(f"\n{'result folder':<52} {'max |diff|':>11}  check", flush=True)
    for name, worst, status in rows:
        print(f"{name:<52} {worst:>11.2e}  {status}", flush=True)
    return failed


def figures(root, out):
    out.mkdir(parents=True, exist_ok=True)
    env = dict(_env(), CHARGEWIN_ROOT=str(root), CHARGEWIN_FIGURES=str(out), MPLBACKEND="Agg",
               PYTHONPATH=str(PKG / "figures"))
    for n, name in enumerate(["datasets", "leakage", "overview", "transfer", "competitors", "fusion", "intervals"], 1):
        subprocess.run([sys.executable, "-B", str(PKG / "figures" / f"make_fig_{name}.py")], cwd=out, env=env,
                       check=True, stdout=subprocess.DEVNULL)
        shutil.copyfile(out / f"Figure_{name}.pdf", out / f"Figure_{n}.pdf")
    print(f"figures written to {out}", flush=True)


def run(tier, data_dir, out_dir, offline=False):
    from chargewin import data, numbers
    start = time.monotonic()
    out_dir = Path(out_dir or REPO / "runs" / tier).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if tier == "paper":
        numbers.check(PAPER)
        figures(PAPER, out_dir / "figures")
        print(f"PASS: paper ({time.monotonic() - start:.0f} s)", flush=True)
        return
    log = out_dir / "logs"
    log.mkdir(exist_ok=True)
    if tier == "quick":
        data.quick_hust(out_dir / "data", offline)
        steps = [s for s in STEPS if s["id"] in QUICK_STEPS]
        for s in steps:
            run_step(s, out_dir, log)
        expected = json.loads((REPO / "expected" / "quick.json").read_text(encoding="utf-8"))
        failed = []
        print(f"\n{'quantity':<46} {'value':>12} {'expected':>12}", flush=True)
        for key, target in expected["values"].items():
            value = numbers.lookup(out_dir, key)
            ok = abs(value - target) <= expected["tolerance"]
            label = key.split("/")[1].split("_")[0] + " " + key.split(":", 1)[1]
            print(f"{label:<46} {value:>12.6f} {target:>12.6f} {'' if ok else 'FAIL'}", flush=True)
            if not ok:
                failed.append(key)
        if failed:
            raise RuntimeError("values differ from expected/quick.json: " + ", ".join(failed))
        print(f"PASS: quick ({time.monotonic() - start:.0f} s)", flush=True)
        return
    tiers = ["main"] if tier == "main" else ["main", "full"]
    steps = [s for s in STEPS if s["tier"] in tiers]
    datasets = sorted({d for s in steps for d in s["data"]})
    data_dir = Path(data_dir).resolve()
    data.prepare(datasets, data_dir, offline)
    _link(data_dir, out_dir / "data")
    if tier == "full":
        for rel in SEEDED:
            if not (out_dir / rel).exists():
                shutil.copytree(PAPER / rel, out_dir / rel)
    for s in steps:
        run_step(s, out_dir, log)
    if tier == "full":
        calce(out_dir, log)
    failed = check_against_paper(steps, out_dir, TOL)
    figures(out_dir, out_dir / "figures")
    if failed:
        for name, problems in failed:
            print(f"FAIL {name}: " + "; ".join(problems), flush=True)
        raise RuntimeError(f"{len(failed)} result folders differ from the paper")
    print(f"PASS: {tier} ({(time.monotonic() - start) / 3600:.1f} h)", flush=True)


def calce(root, log):
    """CALCE illustration: the raw files hold MATLAB tables, so the numeric export needs MATLAB."""
    from chargewin.pipeline import r
    num = root / "tables" / "calce_numeric"
    if not (num / "PL21_numeric.mat").exists():
        matlab = shutil.which("matlab")
        if not matlab:
            print("skip CALCE illustration (needs MATLAB for the numeric export of the raw files)", flush=True)
            return
        src = root / "tables" / "calce_raw"
        import zipfile
        src.mkdir(parents=True, exist_ok=True)
        for z, cell in (("SOC_20-80_2C.zip", "PL17"), ("SOC_20-80_HalfC.zip", "PL21")):
            with zipfile.ZipFile(root / "data" / "calce" / z) as f:
                name = next(n for n in f.namelist() if n.endswith(f"/{cell}.mat"))
                (src / f"{cell}.mat").write_bytes(f.read(name))
        num.mkdir(parents=True, exist_ok=True)
        cmd = f"addpath('{(REPO / 'matlab').as_posix()}'); export_calce_numeric('{src.as_posix()}', '{num.as_posix()}')"
        subprocess.run([matlab, "-batch", cmd], check=True)
    for s in [dict(id=f"C06w{w}", script="c06_calce_genuine_partial.py",
                   args=[".", r(f"C06_CALCE_GENUINE_PARTIAL_2026-10-22_w{w}"), str(w)],
                   out=r(f"C06_CALCE_GENUINE_PARTIAL_2026-10-22_w{w}"), tier="full") for w in (600, 1800)] + [
              dict(id="C52", script="c52_calce_leak_effect.py", args=[".", r("C52_CALCE_LEAK_EFFECT_2026-10-24_v1")],
                   out=r("C52_CALCE_LEAK_EFFECT_2026-10-24_v1"), tier="full"),
              dict(id="C83", script="c83_figure_data.py", args=["."], out=r("C83_FIGURE_DATA_2026-10-29"), tier="full")]:
        run_step(s, root, log)
