# Using orbitcache with asyncio

The core cache is synchronous and safe to call from async code because no
operation blocks on I/O. For memoizing coroutines use `@cache.memoize_async`,
which awaits the wrapped coroutine once and stores the result; concurrent
callers of the same key await a single in-flight computation instead of
racing. Do not share one cache across event loops.
