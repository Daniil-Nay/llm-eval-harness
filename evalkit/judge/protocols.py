"""Judging protocols.

A judge is another model with its own failure modes, so every protocol here
returns raw records (persistable, replayable) instead of a bare score, and a
verdict that fails to parse is counted, never silently dropped from the
denominator.
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

_PAIR_RE = re.compile(r"\b(1|2|tie)\b", re.IGNORECASE)
_POINT_RE = re.compile(r"\b([1-5])\b")


def parse_pairwise(text: str) -> str | None:
    m = _PAIR_RE.search(text.strip()[:20])
    return m.group(1).lower() if m else None


def parse_pointwise(text: str) -> int | None:
    m = _POINT_RE.search(text.strip()[:10])
    return int(m.group(1)) if m else None


def pairwise_with_swap(client, question: str, answer_a: str, answer_b: str) -> dict:
    """Judge the pair in both orders and return a raw record.

    The swap is not optional decoration: without it a position-biased judge
    produces confident, stable, wrong rankings. `agrees` is True only when the
    two orderings name the same underlying answer.
    """
    v_ab = parse_pairwise(client.chat(
        [{"role": "user", "content": PAIRWISE_PROMPT.format(
            question=question, first=answer_a, second=answer_b)}]))
    v_ba = parse_pairwise(client.chat(
        [{"role": "user", "content": PAIRWISE_PROMPT.format(
            question=question, first=answer_b, second=answer_a)}]))

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
        "verdict_ab": v_ab,
        "verdict_ba": v_ba,
        "winner": winner,        # "a" | "b" | "tie" | "unstable" | None
        "agrees": agrees,        # None when either verdict failed to parse
        "skip": None if (v_ab and v_ba) else "judge_parse",
    }
