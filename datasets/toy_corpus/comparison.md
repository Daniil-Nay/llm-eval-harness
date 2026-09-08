# When not to use orbitcache

If several replicas must see the same cached value, use a shared store such as
Redis instead - orbitcache is per-process by design. If values are large and
rarely reused, caching wastes memory: measure the hit rate before assuming a
cache helps. And a database query that is already indexed and fast gains
nothing from an extra caching layer in front of it.
