import ast
import json
from pathlib import Path

from chargewin import data, numbers
from chargewin.compare import compare
from chargewin.pipeline import STEPS

REPO = Path(__file__).resolve().parents[1]
STEP_DIR = REPO / "chargewin" / "steps"


def test_every_step_script_exists_and_parses():
    for s in STEPS:
        ast.parse((STEP_DIR / s["script"]).read_text(encoding="utf-8"))


def test_steps_only_read_earlier_outputs():
    made = set()
    for s in STEPS:
        for a in s["args"]:
            if a.startswith(("results/", "tables/")) and a != s["out"]:
                assert a in made, f"{s['id']} reads {a} before it is written"
        made.add(s["out"])


def test_step_ids_are_unique():
    ids = [s["id"] for s in STEPS]
    assert len(ids) == len(set(ids))


def test_manifest_is_complete():
    for item in data.MANIFEST["files"]:
        assert item["size"] > 0 and len(item["md5"]) == 32 and item["page"].startswith("https://")
    assert sorted(data.MANIFEST["quick_hust_md5"]) == sorted(data.QUICK_HUST_CELLS)


def test_paper_numbers_agree_with_shipped_results():
    numbers.check(REPO / "paper")


def test_quick_reference_values_exist():
    expected = json.loads((REPO / "expected" / "quick.json").read_text(encoding="utf-8"))
    assert expected["values"] and expected["tolerance"] > 0


def test_compare_of_identical_folders_is_zero():
    ref = REPO / "paper" / "results" / "C11_UCONN_LOGO_2026-10-23_v1"
    worst, problems = compare(ref, ref)
    assert worst == 0.0 and not problems


def test_rounding_rule():
    assert numbers.rounds_to(0.0819, "0.082")
    assert not numbers.rounds_to(0.0826, "0.082")
