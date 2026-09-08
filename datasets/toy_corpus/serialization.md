# Serialization backends

By default values are stored as live Python objects, which is the fastest
option and shares mutable state. Set `serializer="json"` to store deep copies
as JSON strings - slower, but immune to accidental mutation and measurable in
bytes. The pickle backend exists for objects JSON cannot express; avoid it for
untrusted data because unpickling runs arbitrary code.
