# First steps

Create a cache with a capacity bound and start storing values:

    from orbitcache import Cache
    cache = Cache(max_items=10_000)
    cache.set("user:42", profile)
    hit = cache.get("user:42")

`get` returns None on a miss. The decorator form `@cache.memoize` wraps a
function so repeated calls with the same arguments skip the body entirely.
