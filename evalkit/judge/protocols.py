"""Judging protocols.

A judge is another model with its own failure modes, so every protocol here
returns raw records (persistable, replayable) instead of a bare score, and a
verdict that fails to parse is reported as a counted skip. Records keep the
judge's raw reply text, so analytics can be recomputed offline and parse
failures can be inspected directly.
"""

import re

PAIRWISE_PROMPT = (
    "Question: {question}\n\n"
    "Answer 1:\n{first}\n\n"
    "Answer 2:\n{second}\n\n"
    "Which answer is better? Reply with exactly one token: 1, 2, or tie."
)

POINTWISE_PROMPT = (
    "Question: {question}\n\n"
    "Answer:\n{answer}\n\n"
    "Rate the answer's quality on a 1-5 integer scale, where 1 is useless and "
    "5 is excellent. Reply with the digit only."
)

RUBRIC_PROMPT = (
    "Question: {question}\n\n"
    "Answer:\n{answer}\n\n"
    "Check the answer against each criterion below. Reply with one line per "
    "criterion, exactly in the form `<criterion>: yes` or `<criterion>: no`, "
    "nothing else.\n\nCriteria:\n{criteria}"
)

_PAIR_RE = re.compile(r"\b(1|2|tie)\b", re.IGNORECASE)
_POINT_RE = re.compile(r"\b([1-5])\b")


def parse_pairwise(text: str) -> str | None:
    m = _PAIR_RE.search(text.strip()[:20])
    return m.group(1).lower() if m else None


def parse_pointwise(text: str) -> int | None:
    m = _POINT_RE.search(text.strip()[:10])
    return int(m.group(1)) if m else None


def parse_rubric(text: str, criteria: list[str]) -> dict[str, bool | None]:
    """Per-criterion yes/no; a criterion the judge did not answer is None."""
    verdicts: dict[str, bool | None] = {c: None for c in criteria}
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, _, value = line.partition(":")
        name = name.strip().strip("`*- ").lower()
        value = value.strip().lower()
        for c in criteria:
            if name == c.lower() and value in ("yes", "no"):
                verdicts[c] = value == "yes"
    return verdicts


def pairwise_with_swap(client, question: str, answer_a: str, answer_b: str,
                       max_tokens: int = 600) -> dict:
    """Judge the pair in both orders and return a raw record.

    Without the swap, a position-biased judge produces rankings that look
    stable and are wrong. `agrees` is True only when the two orderings name
    the same underlying answer.
    """
    raw_ab = client.chat(
        [{"role": "user", "content": PAIRWISE_PROMPT.format(
            question=question, first=answer_a, second=answer_b)}],
        max_tokens=max_tokens)
    raw_ba = client.chat(
        [{"role": "user", "content": PAIRWISE_PROMPT.format(
            question=question, first=answer_b, second=answer_a)}],
        max_tokens=max_tokens)
    v_ab = parse_pairwise(raw_ab)
    v_ba = parse_pairwise(raw_ba)

    winner = None
    agrees = None
    if v_ab and v_ba:
        map_ab = {"1": "a", "2": "b", "tie": "tie"}
        map_ba = {"1": "b", "2": "a", "tie": "tie"}
        first, second = map_ab[v_ab], map_ba[v_ba]
        agrees = first == second
        winner = first if agrees else "unstable"
    return {
        "question": question,
        "raw_ab": raw_ab,
        "raw_ba": raw_ba,
        "verdict_ab": v_ab,
        "verdict_ba": v_ba,
        "winner": winner,        # "a" | "b" | "tie" | "unstable" | None
        "agrees": agrees,        # None when either verdict failed to parse
        "skip": None if (v_ab and v_ba) else "judge_parse",
    }


def pointwise(client, question: str, answer: str, max_tokens: int = 600) -> dict:
    """Score one answer on a 1-5 scale; a failed parse is a counted skip."""
    raw = client.chat(
        [{"role": "user", "content": POINTWISE_PROMPT.format(
            question=question, answer=answer)}],
        max_tokens=max_tokens)
    score = parse_pointwise(raw)
    return {
        "question": question,
        "raw": raw,
        "score": score,          # 1..5 | None
        "skip": None if score else "judge_parse",
    }


def rubric_based(client, question: str, answer: str, criteria: list[str],
                 max_tokens: int = 600) -> dict:
    """Check one answer against named yes/no criteria.

    Rubrics narrow what "better" means: instead of one opaque preference the
    judge answers several small questions a human can audit one by one.
    `skip` is set when any criterion went unanswered. Partial verdicts are
    kept, but the record is flagged.
    """
    raw = client.chat(
        [{"role": "user", "content": RUBRIC_PROMPT.format(
            question=question, answer=answer,
            criteria="\n".join(f"- {c}" for c in criteria))}],
        max_tokens=max_tokens)
    verdicts = parse_rubric(raw, criteria)
    unanswered = [c for c, v in verdicts.items() if v is None]
    return {
        "question": question,
        "raw": raw,
        "verdicts": verdicts,    # criterion -> True | False | None
        "skip": "judge_parse" if unanswered else None,
    }
