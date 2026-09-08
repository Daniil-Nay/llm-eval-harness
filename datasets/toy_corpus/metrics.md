# Telemetry and hit rate

`cache.stats()` returns hits, misses, evictions and the current item count.
Hit rate is hits divided by lookups; a rate below roughly 0.5 usually means
the working set does not fit the bound or the TTL is shorter than the reuse
interval. A hook can be registered with `on_evict` to forward eviction events
to your metrics pipeline, one call per evicted entry.
