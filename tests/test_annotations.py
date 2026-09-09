import json
from pathlib import Path

ANN = Path(__file__).parent.parent / "datasets" / "annotations"
PARSE_FAILS = (Path(__file__).parent.parent / "experiments" / "parse_failures"
               / "results" / "2026-09-09.json")


def _load(name):
    return [json.loads(line) for line in
            (ANN / name).read_text(encoding="utf-8").splitlines() if line]


def test_annotator_b_covers_every_item_exactly_once():
    item_ids = [r["id"] for r in _load("items.jsonl")]
    grade_ids = [r["id"] for r in _load("annotator_b.jsonl")]
    assert len(item_ids) == 40
    assert grade_ids == item_ids


def test_grades_are_on_the_rubric_scale():
    grades = [r["grade"] for r in _load("annotator_b.jsonl")]
    assert all(g in (0, 1, 2, 3) for g in grades)
    # the RUBRIC.md caveat: this particular set is compressed into 2s and 3s
    assert set(grades) == {2, 3}


def test_parse_failures_analysis_matches_raw_records():
    payload = json.loads(PARSE_FAILS.read_text(encoding="utf-8"))
    for condition, stored in payload["analysis"].items():
        rows = [r for r in payload["records"] if r["condition"] == condition]
        parsed = [r for r in rows if r["score"] is not None]
        assert stored["attempted"] == len(rows) == 20
        assert stored["parsed"] == len(parsed)
        assert stored["skipped"] == len(rows) - len(parsed)
    assert payload["analysis"]["normal"]["skip_rate"] == 0.0
    assert payload["analysis"]["starved"]["skip_rate"] == 1.0
    # every skip ships its evidence: the raw reply text is present (may be "")
    for r in payload["records"]:
        assert "raw" in r
