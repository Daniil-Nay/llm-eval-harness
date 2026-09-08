# Snapshots and warm starts

`cache.snapshot(path)` writes the current entries to disk; `Cache.restore(path)`
loads them at startup, so a redeployed service begins warm instead of paying
the miss storm. Snapshots respect TTLs: entries that expired between snapshot
and restore are dropped during load. Snapshotting is stop-the-world for the
cache, so schedule it off the hot path.
