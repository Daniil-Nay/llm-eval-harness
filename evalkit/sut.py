"""System-under-test protocol and a keyword baseline.

Anything that can retrieve documents for a query can be evaluated by this
harness: a real RAG service over HTTP, or the toy baseline below. The
baseline is deliberately naive (token overlap scoring). It exists so the whole
pipeline runs offline in CI, and so there is something honest to beat.
"""

from typing import Protocol

from .datasets import normalize


class SystemUnderTest(Protocol):
    def retrieve(self, query: str, k: int) -> list[str]:
        """Return up to k document ids, best first."""
        ...


class KeywordBaseline:
    """Scores documents by weighted token overlap with the query.

    Rare tokens weigh more (a crude idf): overlap on "the" proves nothing,
    overlap on "eviction" does. No stemming, no embeddings. It is the floor to
    beat rather than a retrieval recommendation.

    `weighting="uniform"` turns the idf weighting off (every shared token
    counts as 1.0). It exists as a legitimately worse variant: the shipped
    runs/ directory diffs it against the default so there is a real, measured
    regression to look at without calling any API.
    """

    def __init__(self, corpus: dict[str, str], weighting: str = "idf"):
        if weighting not in ("idf", "uniform"):
            raise ValueError(f"unknown weighting {weighting!r}")
        self.docs = {doc_id: set(normalize(text).split())
                     for doc_id, text in corpus.items()}
        n_docs = len(self.docs) or 1
        df: dict[str, int] = {}
        for tokens in self.docs.values():
            for tok in tokens:
                df[tok] = df.get(tok, 0) + 1
        if weighting == "uniform":
            self._weight = {tok: 1.0 for tok in df}
        else:
            # weight ~ how selective the token is across the corpus
            self._weight = {tok: 1.0 - (count - 1) / n_docs for tok, count in df.items()}

    def retrieve(self, query: str, k: int) -> list[str]:
        q_tokens = set(normalize(query).split())
        scored = []
        for doc_id, tokens in self.docs.items():
            score = sum(self._weight.get(tok, 1.0) for tok in q_tokens & tokens)
            if score > 0:
                scored.append((score, doc_id))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [doc_id for _score, doc_id in scored[:k]]
