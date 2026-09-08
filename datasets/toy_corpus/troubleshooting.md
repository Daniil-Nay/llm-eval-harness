# Troubleshooting

Memory keeps growing: check that a TTL is set and remember expiry is lazy;
call purge() periodically. Hit rate near zero after deploy: the cache starts
cold - consider snapshots for warm starts. KeyError inside memoize: the
wrapped function's arguments must be hashable. Stale values served: entries
are copies only under the json serializer; live objects reflect later
mutations.
