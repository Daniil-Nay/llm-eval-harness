"""Position bias in an LLM-as-judge: measure, persist, replay.

For each question we generate two comparable answers, then ask the judge
"which is better" in BOTH orders. A consistent judge picks the same answer
regardless of slot; any change of winner when only the order is swapped is
position bias.

Improvements over the original 2026-06-09 run:
  * generator and judge come from DIFFERENT model families, so the
    self-preference confound (a model judging its own text) is removed;
  * every verdict is persisted to results/<date>.json, so the headline numbers
    are recomputed from raw records instead of typed by hand;
  * --replay <file> recomputes all analytics offline, no API key needed;
  * skipped pairs are reported alongside the valid ones;
  * proportions come with Wilson 95% confidence intervals.

Env (OpenAI-compatible endpoints):
  LLM_BASE_URL, LLM_API_KEY                 required for a live run
  GEN_MODEL   (default qwen/qwen3.8-27b)    writes the answer pairs
  JUDGE_MODEL (default openai/gpt-oss-120b) judges them
"""

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from questions import QUESTIONS  # noqa: E402

GEN_MODEL = os.environ.get("GEN_MODEL", "qwen/qwen3.8-27b")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-oss-120b")


def call(messages, model, max_tokens, temperature=0.0, retries=6):
    base = os.environ["LLM_BASE_URL"].rstrip("/")
    key = os.environ["LLM_API_KEY"]
    for _attempt in range(retries):
        r = requests.post(base + "/chat/completions",
                          headers={"Authorization": "Bearer " + key},
                          json={"model": model, "messages": messages,
                                "max_tokens": max_tokens, "temperature": temperature},
                          timeout=90)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"].get("content") or ""
        if r.status_code == 429:
            wait = 8.0
            m = re.search(r"try again in ([\d.]+)s", r.text or "")
            if m:
                wait = min(float(m.group(1)) + 0.5, 30.0)
            time.sleep(wait)
            continue
        raise RuntimeError(f"LLM {r.status_code}: {r.text[:200]}")
    raise RuntimeError("LLM 429: retries exhausted")


def gen_pair(question):
    out = call([{"role": "user", "content":
                 f"Write TWO different, plausible, roughly equally good short answers "
                 f"(2-3 sentences each) to the question below. Label them exactly:\n"
                 f"A: <first answer>\nB: <second answer>\n\nQuestion: {question}"}],
               GEN_MODEL, max_tokens=500, temperature=0.7)
    m = re.search(r"A:\s*(.+?)\nB:\s*(.+)", out, re.S)
    if not m:
        return None
    a, b = m.group(1).strip(), m.group(2).strip()
    return (a, b) if a and b else None


def judge(question, first, second):
    # max_tokens is generous on purpose: reasoning models spend tokens on hidden
    # thinking from the SAME budget, and with max_tokens=3 the visible answer never
    # arrives (finish_reason=length). Cost per verdict stays tiny either way.
    out = call([{"role": "user", "content":
                 f"Question: {question}\n\nAnswer 1:\n{first}\n\nAnswer 2:\n{second}\n\n"
                 f"Which answer is better? Reply with only one character: 1 or 2."}],
               JUDGE_MODEL, max_tokens=600)
    m = re.search(r"[12]", out.strip()[:10])
    return m.group(0) if m else None


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion; returns (lo, hi)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def analyze(records):
    attempted = len(records)
    valid = [r for r in records if r["verdict_ab"] and r["verdict_ba"]]
    skipped = attempted - len(valid)
    # map both orders back to the underlying answer so a "tie" verdict is
    # handled the same way evalkit.judge.bias does (tie/tie = consistent)
    map_ab = {"1": "first", "2": "second", "tie": "tie"}
    map_ba = {"1": "second", "2": "first", "tie": "tie"}
    flips = sum(1 for r in valid
                if map_ab[r["verdict_ab"]] != map_ba[r["verdict_ba"]])
    # slot preference: every parsed non-tie verdict counts, including the
    # surviving half of a partially skipped pair (same rule as evalkit)
    second_slot = judgments = 0
    for r in records:
        for v in (r["verdict_ab"], r["verdict_ba"]):
            if v in ("1", "2"):
                judgments += 1
                second_slot += v == "2"
    n = len(valid)
    flip_lo, flip_hi = wilson(flips, n)
    s_lo, s_hi = wilson(second_slot, judgments)
    return {
        "attempted": attempted,
        "valid": n,
        "skipped": skipped,
        "flips": flips,
        "flip_rate": round(flips / n, 4) if n else None,
        "flip_rate_ci95": [round(flip_lo, 4), round(flip_hi, 4)],
        "second_slot_picks": second_slot,
        "judgments": judgments,
        "second_slot_rate": round(second_slot / judgments, 4) if judgments else None,
        "second_slot_ci95": [round(s_lo, 4), round(s_hi, 4)],
    }


def report(payload):
    a = payload["analysis"]
    print(f"\ngenerator: {payload['gen_model']}   judge: {payload['judge_model']}")
    print(f"pairs attempted {a['attempted']} | valid {a['valid']} | skipped {a['skipped']}")
    if not a["valid"]:
        print("nothing to analyze")
        return
    print(f"winner FLIPPED on swap : {a['flips']}/{a['valid']} = {a['flip_rate']:.0%}"
          f"   (95% CI {a['flip_rate_ci95'][0]:.0%}-{a['flip_rate_ci95'][1]:.0%})")
    print(f"picked the SECOND slot : {a['second_slot_picks']}/{a['judgments']}"
          f" = {a['second_slot_rate']:.0%}"
          f"   (95% CI {a['second_slot_ci95'][0]:.0%}-{a['second_slot_ci95'][1]:.0%})")
    print("\nCaveat: small sample, single judge model. Illustrative, not a benchmark.")
    print("Not run: verbosity bias, self-preference (needs padded-pair and cross-family setups).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", metavar="FILE", help="recompute analytics from a saved raw file")
    ap.add_argument("--out", default=None, help="where to write raw results")
    args = ap.parse_args()

    if args.replay:
        payload = json.loads(Path(args.replay).read_text(encoding="utf-8"))
        payload["analysis"] = analyze(payload["records"])
        report(payload)
        return

    records = []
    for i, q in enumerate(QUESTIONS, 1):
        pair = gen_pair(q)
        if not pair:
            records.append({"question": q, "a": None, "b": None,
                            "verdict_ab": None, "verdict_ba": None, "skip": "gen_parse"})
            print(f"[{i:02d}/20] skip (generation did not parse)")
            continue
        a, b = pair
        v1 = judge(q, a, b)
        time.sleep(0.4)
        v2 = judge(q, b, a)
        time.sleep(0.4)
        records.append({"question": q, "a": a, "b": b,
                        "verdict_ab": v1, "verdict_ba": v2,
                        "skip": None if (v1 and v2) else "judge_parse"})
        flip = (v1 and v2) and ((v1 == "1") != (v2 == "2"))
        print(f"[{i:02d}/20] ab={v1} ba={v2}{'  FLIP' if flip else ''}")

    payload = {
        "date": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gen_model": GEN_MODEL,
        "judge_model": JUDGE_MODEL,
        "judge_prompt": "Which answer is better? Reply with only one character: 1 or 2.",
        "records": records,
    }
    payload["analysis"] = analyze(records)
    out = Path(args.out) if args.out else HERE / "results" / (dt.date.today().isoformat() + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nraw records -> {out}")
    report(payload)


if __name__ == "__main__":
    main()
