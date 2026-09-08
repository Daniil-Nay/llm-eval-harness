# Key sharding

A single big cache contends on one lock. `ShardedCache(shards=16)` hashes each
key to one of several independent shards, each with its own lock and bound,
which spreads contention across threads. Sharding changes eviction accounting:
bounds apply per shard, so a hot shard can evict while a cold one sits half
empty. Pick the shard count once - resharding rehashes every key.
