"""Generate the toy corpus and golden set.

The corpus documents a fictional in-process caching library ("orbitcache") -
fictional on purpose: the dataset is a test fixture with known-good structure,
not knowledge anyone should learn from. Queries are phrased in different words
than the docs so the leakage detector stays quiet on the clean set.
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
CORPUS = ROOT / "datasets" / "toy_corpus"
GOLDEN = ROOT / "datasets" / "toy_golden.jsonl"

DOCS = {
    "overview": """# What orbitcache is

Orbitcache is an in-process caching library for Python services. It keeps hot
values in memory next to your code, so a lookup costs a dictionary access
instead of a network round trip. It is not a distributed cache: every worker
process holds its own copy, and workers never talk to each other. Typical uses
are memoizing expensive computations, caching rendered fragments, and smoothing
over slow upstream APIs.""",

    "installation": """# Installing

Orbitcache supports Python 3.10 and newer and has zero runtime dependencies.
Install from PyPI with `pip install orbitcache`. For development extras
(benchmark helpers and the test suite) use `pip install orbitcache[dev]`.
On Alpine-based Docker images no compiler is needed because the wheel is pure
Python.""",

    "quickstart": """# First steps

Create a cache with a capacity bound and start storing values:

    from orbitcache import Cache
    cache = Cache(max_items=10_000)
    cache.set("user:42", profile)
    hit = cache.get("user:42")

`get` returns None on a miss. The decorator form `@cache.memoize` wraps a
function so repeated calls with the same arguments skip the body entirely.""",

    "eviction": """# Eviction policies

When the cache is full, something has to go. Orbitcache ships three policies:
LRU evicts the least recently used entry and is the default; LFU evicts the
least frequently used one and suits skewed access patterns; FIFO evicts the
oldest insert regardless of use. The policy is chosen per cache instance via
`Cache(policy="lfu")`. Eviction runs inline on insert, so a full cache never
grows past its bound.""",

    "ttl": """# Time-to-live semantics

Every entry may carry a TTL: `cache.set(key, value, ttl=300)` keeps the value
for five minutes. Expiry is lazy - an expired entry is removed on the next
access, not by a background thread. That keeps the library dependency-free but
means memory is reclaimed only when keys are touched. Call `cache.purge()`
to sweep all expired entries eagerly, for example from a periodic job.""",

    "serialization": """# Serialization backends

By default values are stored as live Python objects, which is the fastest
option and shares mutable state. Set `serializer="json"` to store deep copies
as JSON strings - slower, but immune to accidental mutation and measurable in
bytes. The pickle backend exists for objects JSON cannot express; avoid it for
untrusted data because unpickling runs arbitrary code.""",

    "sharding": """# Key sharding

A single big cache contends on one lock. `ShardedCache(shards=16)` hashes each
key to one of several independent shards, each with its own lock and bound,
which spreads contention across threads. Sharding changes eviction accounting:
bounds apply per shard, so a hot shard can evict while a cold one sits half
empty. Pick the shard count once - resharding rehashes every key.""",

    "metrics": """# Telemetry and hit rate

`cache.stats()` returns hits, misses, evictions and the current item count.
Hit rate is hits divided by lookups; a rate below roughly 0.5 usually means
the working set does not fit the bound or the TTL is shorter than the reuse
interval. A hook can be registered with `on_evict` to forward eviction events
to your metrics pipeline, one call per evicted entry.""",

    "async-api": """# Using orbitcache with asyncio

The core cache is synchronous and safe to call from async code because no
operation blocks on I/O. For memoizing coroutines use `@cache.memoize_async`,
which awaits the wrapped coroutine once and stores the result; concurrent
callers of the same key await a single in-flight computation instead of
racing. Do not share one cache across event loops.""",

    "persistence": """# Snapshots and warm starts

`cache.snapshot(path)` writes the current entries to disk; `Cache.restore(path)`
loads them at startup, so a redeployed service begins warm instead of paying
the miss storm. Snapshots respect TTLs: entries that expired between snapshot
and restore are dropped during load. Snapshotting is stop-the-world for the
cache, so schedule it off the hot path.""",

    "configuration": """# Configuration reference

Constructor arguments beat environment variables, which beat built-in
defaults. Recognized variables: ORBITCACHE_MAX_ITEMS (default 100000),
ORBITCACHE_POLICY (lru), ORBITCACHE_DEFAULT_TTL (unset, meaning no expiry).
A value of 0 for max items disables bounding entirely - useful in tests,
dangerous in production.""",

    "troubleshooting": """# Troubleshooting

Memory keeps growing: check that a TTL is set and remember expiry is lazy;
call purge() periodically. Hit rate near zero after deploy: the cache starts
cold - consider snapshots for warm starts. KeyError inside memoize: the
wrapped function's arguments must be hashable. Stale values served: entries
are copies only under the json serializer; live objects reflect later
mutations.""",

    "comparison": """# When not to use orbitcache

If several replicas must see the same cached value, use a shared store such as
Redis instead - orbitcache is per-process by design. If values are large and
rarely reused, caching wastes memory: measure the hit rate before assuming a
cache helps. And a database query that is already indexed and fast gains
nothing from an extra caching layer in front of it.""",
}

GOLDEN_ROWS = [
    # fact
    ("Does every worker share one cache or does each process keep its own?", ["overview"], "fact"),
    ("Which Python versions does the library run on?", ["installation"], "fact"),
    ("What does get return when the key is absent?", ["quickstart"], "fact"),
    ("Which eviction policy is used when none is chosen?", ["eviction"], "fact"),
    ("Is expired data removed by a background thread?", ["ttl"], "fact"),
    ("Why is the pickle backend risky for data from users?", ["serialization"], "fact"),
    ("Do capacity bounds apply per shard or for the whole cache?", ["sharding"], "fact"),
    ("What fields does stats() report?", ["metrics"], "fact"),
    ("Can two event loops share a single cache object?", ["async-api"], "fact"),
    ("Are entries that expired in the meantime loaded back from a snapshot?", ["persistence"], "fact"),
    ("What happens when max items is set to zero?", ["configuration"], "fact"),
    ("Which config source wins when both a constructor argument and an env var are set?", ["configuration"], "fact"),
    ("How many runtime dependencies does the package pull in?", ["installation"], "fact"),
    ("What does a hit rate below one half typically indicate?", ["metrics"], "fact"),
    # procedure
    ("How do I install the development extras?", ["installation"], "procedure"),
    ("How do I memoize a regular function?", ["quickstart"], "procedure"),
    ("How do I pick LFU instead of the default policy?", ["eviction"], "procedure"),
    ("How do I set an expiry of five minutes on a value?", ["ttl"], "procedure"),
    ("How do I force removal of all expired entries right now?", ["ttl"], "procedure"),
    ("How do I switch storage to JSON copies?", ["serialization"], "procedure"),
    ("How do I spread lock contention across threads?", ["sharding"], "procedure"),
    ("How do I forward eviction events to my monitoring?", ["metrics"], "procedure"),
    ("How do I memoize a coroutine?", ["async-api"], "procedure"),
    ("How do I make a redeployed service start with a warm cache?", ["persistence"], "procedure"),
    ("How do I write current entries to a file on disk?", ["persistence"], "procedure"),
    ("How do I bound the number of stored items via environment?", ["configuration"], "procedure"),
    # comparison / judgment
    ("When is a shared store like Redis the better choice?", ["comparison"], "comparison"),
    ("LFU or LRU for a heavily skewed access pattern?", ["eviction"], "comparison"),
    ("Live objects or JSON copies - which protects against mutation?", ["serialization"], "comparison"),
    ("Is adding a cache in front of a fast indexed query worth it?", ["comparison"], "comparison"),
    ("Single big cache versus many shards - what is the trade-off?", ["sharding"], "comparison"),
    # troubleshooting / multi-doc
    ("Memory of my service keeps climbing even though I set TTLs - why?", ["troubleshooting", "ttl"], "troubleshooting"),
    ("After a deploy the hit rate is terrible for several minutes", ["troubleshooting", "persistence"], "troubleshooting"),
    ("I get KeyError from the memoize decorator", ["troubleshooting"], "troubleshooting"),
    ("Cached values change after I mutate the original object", ["troubleshooting", "serialization"], "troubleshooting"),
    ("Concurrent coroutines all recompute the same expensive key", ["async-api"], "troubleshooting"),
    ("A hot shard keeps evicting while others stay half empty", ["sharding"], "troubleshooting"),
    ("Why does my cache never free memory for keys nobody touches?", ["ttl"], "troubleshooting"),
    ("Snapshot pauses my service - what should I do about it?", ["persistence"], "troubleshooting"),
    ("What tells me whether my working set fits the configured bound?", ["metrics"], "troubleshooting"),
]


def main():
    CORPUS.mkdir(parents=True, exist_ok=True)
    for doc_id, text in DOCS.items():
        (CORPUS / f"{doc_id}.md").write_text(text.strip() + "\n", encoding="utf-8")
    with GOLDEN.open("w", encoding="utf-8") as f:
        for query, ids, qtype in GOLDEN_ROWS:
            f.write(json.dumps({"query": query, "relevant_ids": ids, "type": qtype},
                               ensure_ascii=False) + "\n")
    print(f"corpus: {len(DOCS)} docs -> {CORPUS}")
    print(f"golden: {len(GOLDEN_ROWS)} rows -> {GOLDEN}")


if __name__ == "__main__":
    main()
