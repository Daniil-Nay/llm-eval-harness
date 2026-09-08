import pytest

from evalkit.metrics.generation import citation_precision, citation_supported

SOURCE = ("Expiry is lazy - an expired entry is removed on the next access, "
          "not by a background thread.")


def test_supported_quote_survives_cosmetics():
    assert citation_supported("expired entry is removed on the NEXT access", SOURCE)
    assert citation_supported("removed on the next access, not by a background thread!", SOURCE)


def test_fabricated_quote_fails():
    assert not citation_supported("entries are swept by a background thread", SOURCE)


def test_too_short_quote_rejected_even_if_present():
    assert not citation_supported("expired entry", SOURCE)


def test_precision_counts_only_real_pairs():
    sources = {"ttl": SOURCE, "other": "sharding spreads contention across locks"}
    quotes = ["removed on the next access",
              "sharding spreads contention",
              "removed on the next access"]
    cited = ["ttl", "other", "other"]  # third one cites the wrong doc
    assert citation_precision(quotes, sources, cited) == pytest.approx(2 / 3)


def test_precision_unknown_doc_counts_as_unsupported():
    assert citation_precision(["removed on the next access"], {"ttl": SOURCE},
                              ["nonexistent"]) == 0.0


def test_precision_rejects_length_mismatch():
    with pytest.raises(ValueError):
        citation_precision(["a quote here"], {}, ["x", "y"])
