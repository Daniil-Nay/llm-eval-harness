# llm-eval-harness

A small toolkit for evaluating LLM/RAG systems, plus a reproducible experiment
on position bias in LLM-as-judge setups. Stdlib + `requests` only; everything
deterministic runs offline; every proportion ships with a confidence interval.

## The finding

Ask a judge model "which answer is better?" twice, swapping the order of the
two answers between calls. A consistent judge names the same answer both times.

On 20 question/answer pairs (generator `qwen3.8-27b`, judge `gpt-oss-120b`,
different model families on purpose):

| measurement | result | 95% CI |
|---|---|---|
| verdict flipped when order swapped | 4/20 = **20%** | 8–42% |
| second slot picked, across all 40 judgments | 20/40 = **50%** | 35–65% |

An earlier prototype of this experiment (June 2026) reported 35% flips and a
68% preference for the second slot. Those numbers came from a sloppier setup:
the same model family generated and judged the answers, unparseable verdicts
were dropped from the denominator, and the results file was typed by hand.
Fixing the setup — cross-family judge, persisted raw verdicts, counted skips —
shrank the flip rate and made the slot preference disappear entirely.

That is the actual lesson of this repo. The measurement infrastructure changed
the conclusion more than any prompt tweak ever did. And at n=20 the flip-rate
interval spans 8–42%, which is why the harness refuses to print a proportion
without one.

Raw per-pair verdicts live in
[`experiments/position_bias/results/`](experiments/position_bias/results/), and
the headline numbers are recomputed from them by the test suite
(`tests/test_judge.py`) and by `run.py --replay` — nothing is typed by hand
anymore.

```bash
python experiments/position_bias/run.py --replay experiments/position_bias/results/2026-09-08.json
```

## What's inside

```
evalkit/
  metrics/retrieval.py   recall@k, precision@k, MRR, nDCG@k
  metrics/agreement.py   Cohen's kappa, weighted kappa
  metrics/generation.py  citation support / citation precision (no LLM involved)
  stats.py               Wilson interval, bootstrap CI, paired bootstrap diff
  datasets.py            golden-set loading + hygiene: exact/near duplicates,
                         corpus leakage, type skew
  runner.py              run a SUT over a golden set: per-example scores,
                         pinned manifest, diff of two runs
  judge/client.py        OpenAI-compatible chat client (env-configured)
  judge/cache.py         disk cache for judge calls - a rerun costs nothing
  judge/protocols.py     pairwise-with-swap, pointwise, rubric-based;
                         parse-failure = counted skip, raw reply persisted
  judge/bias.py          flip rate + slot preference over persisted records
  sut.py                 SystemUnderTest protocol + keyword baseline
  gate.py                CI regression gate, exit code as the interface
datasets/                toy corpus (13 docs) + golden set (40 queries)
datasets/annotations/    40 answers graded 0-3 by the author against a
                         written rubric - a fixed second annotator
runs/                    baseline vs degraded-variant retrieval runs, diffable
experiments/position_bias/   the experiment above, with raw results
experiments/parse_failures/  the same judge with a starved token budget:
                             skip rate 0/20 at max_tokens=600, 20/20 at 4
```

The toy corpus documents a fictional caching library, so retrieval quality here
measures the harness, never the model's memorized knowledge.

## Working offline

Everything deterministic runs without an API key: the test suite, the gate,
`--replay` on both experiments, and the shipped `runs/` diff. A key is needed
only to produce your own numbers instead of replaying the committed ones.

## Quickstart

```bash
pip install -e .[dev]
pytest -q                  # 48 offline tests, no API key needed
python -m evalkit.gate     # hygiene + recall@5 gate on the toy dataset
```

The gate compares the *lower* bootstrap bound of recall against the threshold,
so it fails on real regressions and stays quiet on sampling noise:

```bash
python -m evalkit.gate --threshold 0.6   # exit 0 if lower CI bound >= 0.6
```

Wire that into CI and a retrieval regression turns the PR red. This repo runs
it on itself — see [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Evaluating your own system

Anything with a `retrieve(query, k) -> list[doc_id]` method satisfies the
`SystemUnderTest` protocol:

```python
from evalkit.datasets import load_golden, validate
from evalkit.metrics.retrieval import recall_at_k
from evalkit.stats import bootstrap_ci

rows = load_golden("datasets/toy_golden.jsonl")
scores = [recall_at_k(my_system.retrieve(r["query"], 5), set(r["relevant_ids"]), 5)
          for r in rows]
print(sum(scores) / len(scores), bootstrap_ci(scores))
```

Run `validate(rows, corpus)` first. A golden set rots quietly — duplicates from
logs, synthetic questions leaking into the indexed corpus, one query type
crowding out the rest — and none of it crashes anything; it just bends every
number computed afterwards.

## Measuring your own judge

```bash
export LLM_BASE_URL=https://api.groq.com/openai/v1   # any OpenAI-compatible endpoint
export LLM_API_KEY=...
export JUDGE_MODEL=openai/gpt-oss-120b
python experiments/position_bias/run.py
```

Or programmatically:

```python
from evalkit.judge.client import JudgeClient
from evalkit.judge.protocols import pairwise_with_swap
from evalkit.judge.bias import flip_rate

records = [pairwise_with_swap(JudgeClient(), q, ans_a, ans_b)
           for q, ans_a, ans_b in my_pairs]
print(flip_rate(records))   # includes skipped count and Wilson CI
```

Pick a judge from a different model family than the system it judges;
otherwise self-preference and position bias are measured as one lump.

## Limitations

- One judge model, n=20 pairs. The intervals above say how little that proves.
- Verbosity bias and self-preference are not measured here (they need
  padded-pair and cross-family setups the experiment doesn't include yet).
- The keyword baseline is a floor for pipeline testing, never a retrieval
  recommendation.

## License

MIT
