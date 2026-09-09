"""Parse failures are real: starve the judge and keep the evidence.

The same pointwise judge rates the same 20 answers twice: once with a normal
token budget and once with max_tokens=4. Reasoning models spend the budget on
hidden thinking first, so the starved condition returns empty or truncated
text and the verdict never arrives. Every raw reply is persisted - the point
of this dataset is that a "skip" is a real, inspectable string, not an
abstraction.

Env: LLM_BASE_URL, LLM_API_KEY, JUDGE_MODEL (same contract as evalkit).
Replay: --replay <file> recomputes the analysis offline.
"""

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))

from evalkit.judge.client import JudgeClient  # noqa: E402
from evalkit.judge.protocols import pointwise  # noqa: E402

SOURCE = HERE.parent / "position_bias" / "results" / "2026-09-08.json"


def analyze(records):
    out = {}
    for condition in ("normal", "starved"):
        rows = [r for r in records if r["condition"] == condition]
        parsed = [r for r in rows if r["score"] is not None]
        out[condition] = {
            "attempted": len(rows),
            "parsed": len(parsed),
            "skipped": len(rows) - len(parsed),
            "skip_rate": round(1 - len(parsed) / len(rows), 4) if rows else None,
            "mean_score": round(sum(r["score"] for r in parsed) / len(parsed), 3)
            if parsed else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", metavar="FILE")
    args = ap.parse_args()

    if args.replay:
        payload = json.loads(Path(args.replay).read_text(encoding="utf-8"))
        payload["analysis"] = analyze(payload["records"])
        print(json.dumps(payload["analysis"], indent=1))
        return

    pairs = json.loads(SOURCE.read_text(encoding="utf-8"))["records"]
    client = JudgeClient()
    records = []
    for i, rec in enumerate(pairs, 1):
        for condition, budget in (("normal", 600), ("starved", 4)):
            r = pointwise(client, rec["question"], rec["a"], max_tokens=budget)
            r["condition"] = condition
            r["max_tokens"] = budget
            records.append(r)
            time.sleep(0.4)
        print(f"[{i:02d}/{len(pairs)}] normal={records[-2]['score']} "
              f"starved={records[-1]['score']}")

    payload = {
        "date": dt.datetime.now(dt.timezone.utc).isoformat(),
        "judge_model": client.model,
        "records": records,
    }
    payload["analysis"] = analyze(records)
    out = HERE / "results" / (dt.date.today().isoformat() + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(json.dumps(payload["analysis"], indent=1))
    print(f"raw records -> {out}")


if __name__ == "__main__":
    main()
