# =============================================================================
# JLD2 Backend for FileCache
# =============================================================================
#
# Provides persistent key-value caching via JLD2 files.
# Loaded automatically when `using JLD2` is called alongside VisualGeometryToolkitFeatures.
#
# =============================================================================

module VisualGeometryToolkitFeaturesJLD2Ext

using JLD2: JLD2, jldopen
import VisualGeometryToolkitFeatures: FileCache, open_cache

"""
    open_cache(path::AbstractString) -> FileCache or nothing

Open a JLD2 file for read/append. Returns `nothing` if the file
cannot be opened (e.g., corrupted, permissions).
"""
function open_cache(path::AbstractString)
    return try
        FileCache(jldopen(path, "a+"))
    catch e
        @warn "Cache unavailable, proceeding without caching" path exception = e
        nothing
    end
end

# `f::Union{Function, Type}` as in Base's `get!(f, dict, key)` methods: an untyped
# `f` would be ambiguous with `get!(::AbstractDict, key, default)`.
"""
    get!(f, cache::FileCache, key::String)

Return `cache.db[key]` if present, otherwise call `f()`, store the result under
`key`, and return it (Base's `get!(f, dict, key)` on the file-backed store).
"""
function Base.get!(f::Union{Function, Type}, cache::FileCache, key::String)
    haskey(cache.db, key) && return cache.db[key]
    result = f()
    cache.db[key] = result
    return result
end

end # module
