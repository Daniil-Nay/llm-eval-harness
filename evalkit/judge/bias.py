"""Bias analytics over raw pairwise-with-swap records.

Works on persisted records (see experiments/position_bias), so all numbers can
be recomputed offline. Ties are treated as consistent verdicts: a judge that
says "tie" both ways is stable, just undecided.
"""

from ..stats import wilson_interval


def flip_rate(records: list[dict]) -> dict:
    """Share of valid pairs where swapping the order changed the winner."""
    valid = [r for r in records if r.get("verdict_ab") and r.get("verdict_ba")]
    skipped = len(records) - len(valid)
    map_ab = {"1": "first", "2": "second", "tie": "tie"}
    map_ba = {"1": "second", "2": "first", "tie": "tie"}
    flips = sum(1 for r in valid
                if map_ab[r["verdict_ab"]] != map_ba[r["verdict_ba"]])
    n = len(valid)
    lo, hi = wilson_interval(flips, n)
    return {"attempted": len(records), "valid": n, "skipped": skipped,
            "flips": flips, "rate": flips / n if n else None,
            "ci95": (round(lo, 4), round(hi, 4))}


def slot_preference(records: list[dict]) -> dict:
    """How often the judge picked whatever was shown in the SECOND slot,
    across all valid non-tie judgments. 50% means no positional pull."""
    picks = total = 0
    for r in records:
        for verdict, second_token in ((r.get("verdict_ab"), "2"),
                                      (r.get("verdict_ba"), "2")):
            if verdict in ("1", "2"):
                total += 1
                picks += verdict == second_token
    lo, hi = wilson_interval(picks, total)
    return {"second_slot_picks": picks, "judgments": total,
            "rate": picks / total if total else None,
            "ci95": (round(lo, 4), round(hi, 4))}
