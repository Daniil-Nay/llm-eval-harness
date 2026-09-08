"""Deterministic generation-side checks.

Only what can be verified without another LLM lives here. Judge-based scoring
belongs to evalkit.judge; mixing the two makes it too easy to forget which
numbers are measurements and which are opinions of another model.
"""

import re


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def citation_supported(quote: str, source_text: str, min_words: int = 3) -> bool:
    """Does the quoted span actually appear in the cited source?

    Comparison is normalized (case, punctuation, whitespace) so cosmetic
    differences do not fail an honest citation. Quotes shorter than
    `min_words` are rejected outright: two-word "quotes" match almost
    anything and prove nothing.
    """
    q = _normalize(quote)
    if len(q.split()) < min_words:
        return False
    return q in _normalize(source_text)


def citation_precision(quotes: list[str], sources: dict[str, str],
                       cited_ids: list[str]) -> float:
    """Share of (quote, cited doc) pairs where the quote is really in that doc.

    `quotes[i]` is claimed to come from `sources[cited_ids[i]]`. Unknown doc id
    counts as an unsupported citation - pointing at a document that does not
    exist is the worst kind of confident.
    """
    if len(quotes) != len(cited_ids):
        raise ValueError("quotes and cited_ids differ in length")
    if not quotes:
        return 0.0
    good = 0
    for quote, doc_id in zip(quotes, cited_ids, strict=True):
        doc = sources.get(doc_id)
        if doc is not None and citation_supported(quote, doc):
            good += 1
    return good / len(quotes)
