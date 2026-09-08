# Configuration reference

Constructor arguments beat environment variables, which beat built-in
defaults. Recognized variables: ORBITCACHE_MAX_ITEMS (default 100000),
ORBITCACHE_POLICY (lru), ORBITCACHE_DEFAULT_TTL (unset, meaning no expiry).
A value of 0 for max items disables bounding entirely - useful in tests,
dangerous in production.
