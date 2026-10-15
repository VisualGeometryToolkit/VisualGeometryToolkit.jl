# =============================================================================
# Persistent Cache — Abstract API (implemented by JLD2 extension)
# =============================================================================
#
# Provides the `get!(f, cache, key)` get-or-compute pattern (Base's
# `get!(f, dict, key)`) for persistent key-value caching.
# The JLD2 backend is loaded via package extension when JLD2 is available.
#
# Usage:
#   cache = open_cache("features.jld2")  # nothing if file can't be opened
#   result = get!(cache, "key") do       # (no `get!` for a `nothing` cache:
#       expensive_computation()          #  compute directly in that case)
#   end
#   close_cache(cache)
#
# =============================================================================

using SHA: sha256

"""
    FileCache{D}

Wrapper around an open file-backed key-value store.
The concrete backend `D` is determined by the extension (e.g., JLD2).
"""
struct FileCache{D}
    db::D
end

"""
    open_cache(path) -> FileCache or nothing

Open a persistent cache file. Returns `nothing` if `path` is `nothing`
or the file cannot be opened (logs a warning).

Requires a backend extension (e.g., `using JLD2`).
"""
open_cache(::Nothing) = nothing

"""
    close_cache(cache)

Close the cache file handle. No-op for `nothing`.
"""
close_cache(::Nothing) = nothing
close_cache(c::FileCache) = close(c.db)

"""
    cache_key(args...) -> String

Compute a stable hex-encoded SHA-256 cache key from arbitrary arguments.
The caller decides what goes in; this function just hashes.

# Example
```julia
key = cache_key(board_uid, model_hash, cutoff, patch_size)
get!(cache, key) do ... end
```
"""
cache_key(args...) = bytes2hex(sha256(repr(args)))
