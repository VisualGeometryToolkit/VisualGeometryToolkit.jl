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
    CACHE_FORMAT

The layout version of the values a [`FileCache`](@ref) stores, part of every
[`cache_key`](@ref). The backend (JLD2) stores Julia types, so a cached value
whose type's layout changes cannot be read back as the new type: raising this
number makes every earlier entry a miss (recomputed), never a load that fails.
Raise it with any change to the layout of a cached type.

- 1: the initial layout.
- 2 (2026-11): VisualGeometryToolkitCore's `CovMat` / `PrecMat` hold a static
  matrix (`CovMat{N, T, M, C}`; was `CovMat{N, T, L, C}`), so cached uncertain
  detections of the earlier layout do not load.
"""
const CACHE_FORMAT = 2

"""
    cache_key(args...) -> String

Compute a stable hex-encoded SHA-256 cache key from arbitrary arguments and
the cache's layout version ([`CACHE_FORMAT`](@ref)): an entry written under an
earlier layout is a miss. The caller decides what else goes in; this function
just hashes.

# Example
```julia
key = cache_key(board_uid, model_hash, cutoff, patch_size)
get!(cache, key) do ... end
```
"""
cache_key(args...) = bytes2hex(sha256(repr((CACHE_FORMAT, args...))))
