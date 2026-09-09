from evalkit.judge.protocols import pairwise_with_swap, parse_rubric, pointwise, rubric_based


class ScriptedJudge:
    model = "fake"

    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages, **_kw):
        return self.replies.pop(0)


def test_swap_record_keeps_raw_texts():
    rec = pairwise_with_swap(ScriptedJudge(["1", "2"]), "q?", "aaa", "bbb")
    assert rec["raw_ab"] == "1" and rec["raw_ba"] == "2"
    assert rec["winner"] == "a"


def test_pointwise_parses_and_counts_skips():
    ok = pointwise(ScriptedJudge(["4"]), "q?", "answer")
    assert ok["score"] == 4 and ok["skip"] is None
    starved = pointwise(ScriptedJudge([""]), "q?", "answer")
    assert starved["score"] is None and starved["skip"] == "judge_parse"
    assert starved["raw"] == ""  # the raw evidence survives


def test_parse_rubric_matches_criteria_case_insensitively():
    text = "Grounded: yes\n- Complete: NO\nsomething else entirely"
    verdicts = parse_rubric(text, ["grounded", "complete", "concise"])
    assert verdicts == {"grounded": True, "complete": False, "concise": None}


def test_rubric_based_flags_unanswered_criteria():
    judge = ScriptedJudge(["grounded: yes\ncomplete: yes"])
    rec = rubric_based(judge, "q?", "ans", ["grounded", "complete", "concise"])
    assert rec["verdicts"]["grounded"] is True
    assert rec["verdicts"]["concise"] is None
    assert rec["skip"] == "judge_parse"

    judge = ScriptedJudge(["grounded: yes\ncomplete: no\nconcise: yes"])
    rec = rubric_based(judge, "q?", "ans", ["grounded", "complete", "concise"])
    assert rec["skip"] is None
