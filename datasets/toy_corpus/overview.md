# What orbitcache is

Orbitcache is an in-process caching library for Python services. It keeps hot
values in memory next to your code, so a lookup costs a dictionary access
instead of a network round trip. It is not a distributed cache: every worker
process holds its own copy, and workers never talk to each other. Typical uses
are memoizing expensive computations, caching rendered fragments, and smoothing
over slow upstream APIs.
