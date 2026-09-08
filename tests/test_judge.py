import json
from pathlib import Path

import pytest

from evalkit.judge.bias import flip_rate, slot_preference
from evalkit.judge.protocols import pairwise_with_swap, parse_pairwise, parse_pointwise

RESULTS = (Path(__file__).parent.parent
           / "experiments" / "position_bias" / "results" / "2026-09-08.json")


def test_parse_pairwise_tolerates_wrappers():
    assert parse_pairwise("1") == "1"
    assert parse_pairwise("  Answer: 2 ") == "2"
    assert parse_pairwise("TIE") == "tie"
    # verdict buried too deep in chatter is a parse failure, not a guess
    assert parse_pairwise("both are fine, honestly, but if pressed... 1") is None
    assert parse_pairwise("") is None


def test_parse_pointwise():
    assert parse_pointwise("4") == 4
    assert parse_pointwise(" score: 5 ") == 5
    assert parse_pointwise("nine") is None


class FakeJudge:
    """Replays scripted replies; records the prompts it saw."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def chat(self, messages, **_kw):
        self.prompts.append(messages[0]["content"])
        return self.replies.pop(0)


def test_swap_consistent_judge_yields_stable_winner():
    # picks A both times: "1" when A is first, "2" when A is second
    rec = pairwise_with_swap(FakeJudge(["1", "2"]), "q?", "answer a", "answer b")
    assert rec["winner"] == "a"
    assert rec["agrees"] is True
    assert rec["skip"] is None


def test_swap_position_biased_judge_is_flagged_unstable():
    # always answers "1" = always prefers whatever sits in the first slot
    rec = pairwise_with_swap(FakeJudge(["1", "1"]), "q?", "answer a", "answer b")
    assert rec["winner"] == "unstable"
    assert rec["agrees"] is False


def test_swap_actually_swaps_the_order():
    judge = FakeJudge(["1", "1"])
    pairwise_with_swap(judge, "q?", "AAA", "BBB")
    first_call, second_call = judge.prompts
    assert first_call.index("AAA") < first_call.index("BBB")
    assert second_call.index("BBB") < second_call.index("AAA")


def test_swap_unparseable_verdict_is_a_skip_not_a_crash():
    rec = pairwise_with_swap(FakeJudge(["garbage", "2"]), "q?", "a", "b")
    assert rec["winner"] is None
    assert rec["skip"] == "judge_parse"


def test_flip_rate_counts_skips_in_attempted():
    records = [
        {"verdict_ab": "1", "verdict_ba": "2"},   # stable
        {"verdict_ab": "1", "verdict_ba": "1"},   # flip
        {"verdict_ab": "tie", "verdict_ba": "tie"},  # stable (undecided)
        {"verdict_ab": None, "verdict_ba": "2"},  # skip
    ]
    res = flip_rate(records)
    assert res["attempted"] == 4
    assert res["valid"] == 3
    assert res["skipped"] == 1
    assert res["flips"] == 1
    assert res["rate"] == pytest.approx(1 / 3)


def test_slot_preference_on_synthetic_always_second():
    records = [{"verdict_ab": "2", "verdict_ba": "2"}] * 5
    res = slot_preference(records)
    assert res["judgments"] == 10
    assert res["rate"] == 1.0


def test_bias_numbers_reproduce_from_shipped_raw_records():
    """The README numbers must be recomputable from the persisted run."""
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    records = payload["records"]
    stored = payload["analysis"]

    flips = flip_rate(records)
    assert flips["flips"] == stored["flips"]
    assert flips["valid"] == stored["valid"]
    assert flips["rate"] == pytest.approx(stored["flip_rate"])
    assert list(flips["ci95"]) == pytest.approx(stored["flip_rate_ci95"], abs=1e-3)

    slots = slot_preference(records)
    assert slots["second_slot_picks"] == stored["second_slot_picks"]
    assert slots["judgments"] == stored["judgments"]
    assert slots["rate"] == pytest.approx(stored["second_slot_rate"])
