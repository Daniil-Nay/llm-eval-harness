# Eviction policies

When the cache is full, something has to go. Orbitcache ships three policies:
LRU evicts the least recently used entry and is the default; LFU evicts the
least frequently used one and suits skewed access patterns; FIFO evicts the
oldest insert regardless of use. The policy is chosen per cache instance via
`Cache(policy="lfu")`. Eviction runs inline on insert, so a full cache never
grows past its bound.
