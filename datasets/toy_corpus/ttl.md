# Time-to-live semantics

Every entry may carry a TTL: `cache.set(key, value, ttl=300)` keeps the value
for five minutes. Expiry is lazy - an expired entry is removed on the next
access, not by a background thread. That keeps the library dependency-free but
means memory is reclaimed only when keys are touched. Call `cache.purge()`
to sweep all expired entries eagerly, for example from a periodic job.
