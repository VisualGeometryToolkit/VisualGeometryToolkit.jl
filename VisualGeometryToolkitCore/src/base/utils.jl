# =============================================================================
# Core Utilities
# =============================================================================
#
# This module coordinates core utility functionality by including specialized files:
# - geometry_utils.jl:  Geometry utilities (center, ranges, coordinate conversions)
# =============================================================================

# =============================================================================
# Type Utilities
# =============================================================================

"""
    filter_kwargs(T::Type; kwargs...) -> (matched::NamedTuple, dropped::NamedTuple)

Split keyword arguments into those that match fieldnames of type `T` and those that don't.

This is useful for forwarding a subset of kwargs to a struct constructor while forwarding
the remainder to another function.

# Arguments
- `T::Type`: Type whose fieldnames determine which kwargs are "matched"
- `kwargs...`: Keyword arguments to filter

# Returns
A tuple of two `NamedTuple`s:
- `matched`: kwargs whose names are fields of `T`
- `dropped`: kwargs whose names are not fields of `T`

# Examples
```julia
Base.@kwdef struct Config
    width::Int = 100
    height::Int = 100
end

matched, dropped = filter_kwargs(Config; width=200, extra=true)
# matched = (width = 200,)
# dropped = (extra = true,)

config = Config(; matched...)
```
"""
function filter_kwargs(::Type{T}; kwargs...) where {T}
    fields = fieldnames(T)
    matched = NamedTuple(k => v for (k, v) in pairs(kwargs) if k in fields)
    dropped = NamedTuple(k => v for (k, v) in pairs(kwargs) if k ∉ fields)
    return matched, dropped
end

"""
    validate_dir(dir; create=false) -> String

Validate a directory path and optionally create it.

# Arguments
- `dir`: Directory path (String or AbstractString)
- `create::Bool=false`: If true, create the directory (and parents) if it doesn't exist

# Returns
The validated directory path as a String.

# Throws
- `ArgumentError` if `dir` is not a directory and `create=false`
- `ArgumentError` if `dir` exists but is not a directory (e.g., it's a file)

# Examples
```julia
# Validate existing directory
path = validate_dir("/tmp")

# Create directory if needed
path = validate_dir("/tmp/mydir"; create=true)
```
"""
function validate_dir(dir; create::Bool = false)
    path = String(dir)
    if isdir(path)
        return path
    elseif ispath(path)
        throw(ArgumentError("Path exists but is not a directory: $path"))
    elseif create
        mkpath(path)
        return path
    else
        throw(ArgumentError("Directory does not exist: $path"))
    end
end


# =============================================================================
# Include Specialized Utility Modules
# =============================================================================

include("geometry_utils.jl")

# =============================================================================
# saturation_bounds — the pixel value range an observation may trust
# =============================================================================

"""
    saturation_bounds(::Type{T}) -> (lo, hi)
    saturation_bounds(::Type{<:Colorant}) -> (lo, hi)
    saturation_bounds(img::AbstractMatrix) -> (lo, hi)

The open interval of pixel values that are NOT clipped by the sensor or the
encoding, as `Float64`s: `(1e-6, 1 - 1e-6)` for float images, one code inside
`typemin`/`typemax` for integer ones, one `eps` inside for fixed-point ones. A pixel outside it carries no
intensity information (it is saturated) and must not enter a photometric
fit. Colorants defer to their element type; an image to its `eltype`.
"""
saturation_bounds(::Type{T}) where {T <: AbstractFloat} = (1.0e-6, 1.0 - 1.0e-6)
saturation_bounds(::Type{T}) where {T <: Integer} = (Float64(typemin(T)) + 1.0, Float64(typemax(T)) - 1.0)
# Fixed-point (N0f8, N0f16): one representable step inside the encoding's range.
saturation_bounds(::Type{T}) where {T <: Real} =
    (Float64(typemin(T)) + Float64(eps(T)), Float64(typemax(T)) - Float64(eps(T)))
saturation_bounds(::Type{C}) where {C <: Colors.Colorant} = saturation_bounds(eltype(C))
saturation_bounds(img::AbstractMatrix) = saturation_bounds(eltype(img))

# =============================================================================
# Seeds
# =============================================================================

"""
    stable_seed(seed, keys...) -> UInt64

The seed derivation of every random stream in the stack: the first eight bytes
(big-endian) of the SHA-256 of `seed` and `keys` printed as text, separated by
the unit separator `\\x1f`. A fixed algorithm, unlike `Base.hash` (which
changes across Julia versions), and keyed by NAME, unlike `seed + i`: a
scene's draw depends on that scene's name only, so adding, removing or
reordering other scenes leaves it unchanged, and no arithmetic coincidence
gives two key tuples one stream. Keys are told apart by their printed text
(`string`): `2`, `"2"` and `Int32(2)` are one key, and a key whose text holds
`\\x1f` can collide with two keys; integers and plain names are always
distinct. Seed a StableRNGs
`StableRNG` with it (`StableRNG(stable_seed(seed, "scene", i))`), whose
stream is fixed across Julia versions too.

```jldoctest
julia> using VisualGeometryToolkitCore: stable_seed

julia> stable_seed(7, "budget", 2, 1)
0xf5154d7e8f0e4d1c

julia> stable_seed(1000, 1, 3) == stable_seed(0, 2, 3)
false
```
"""
stable_seed(seed::Integer, keys...) =
    reduce((acc, b) -> (acc << 8) | UInt64(b), sha256(join((string(seed), string.(keys)...), '\x1f'))[1:8];
           init = zero(UInt64))

"""
    sha256_hex(s::AbstractString) -> String
    sha256_hex(bytes::AbstractVector{UInt8}) -> String
    sha256_hex(io::IO) -> String

Lower-case hex SHA-256 — the one hash behind every content identity in the
stack (VisualGeometryToolkitIO's `sha256_hex(canonical_json(tree))`) and every
file digest (a pinned model's, a data artifact's). A string is hashed as its
bytes (never opened as a path); a stream is read to its end. A file's digest is
the stream form, `open(sha256_hex, path)`.
"""
sha256_hex(s::AbstractString) = sha256_hex(codeunits(s))
sha256_hex(bytes::AbstractVector{UInt8}) = bytes2hex(sha256(bytes))
sha256_hex(io::IO) = bytes2hex(sha256(io))

"""
    RetiredKeyword()
    refuse_retired_keyword(value, old, new) -> nothing

The stack's one refusal of a retired keyword, never an alias: a function keeps the
retired keyword `old` with the default `RetiredKeyword()` and calls
`refuse_retired_keyword(old_value, :old, successor)` first, which throws
`ArgumentError("the keyword `old` is retired: give `new`")` for any value a caller
supplies and returns `nothing` for the default. `new` names what replaces it (a
keyword, or the argument that now carries it). `LensSpec` refuses `free` this way
(it takes `fixed`), every `find_*`, `ransac` and `locally_optimize` refuses
`local_optimization` (it takes `local_optimization_strategy`), every `refine_*`
and `fit_*` the strategy spellings (the owner's ruling, 2026-11-11); a downstream
package refuses its own retired keywords through the same pair.
"""
struct RetiredKeyword end
"""
    refuse_retired_keyword(value, old, new) -> nothing

`nothing` for [`RetiredKeyword`](@ref)`()`, the retired keyword's default; for any
other `value` an `ArgumentError` naming the retired keyword `old` and its
successor `new`.
"""
refuse_retired_keyword(::RetiredKeyword, _old, _new) = nothing
refuse_retired_keyword(_value, old, new) =
    throw(ArgumentError("the keyword `$old` is retired: give `$new`"))

"""
    require_extension(missing_error, parent::Module, extension::Symbol) -> nothing
    require_extension(missing_error, f, args::Tuple, parent::Module, extension::Symbol)

The stack's one check for a package extension a call needs. `missing_error` is a
zero-argument function returning the exception to throw when the extension is not
loaded: the caller's own message, naming what to install and load (`using X`),
built only on that path.

- `require_extension(missing_error, parent, extension)`: `nothing` when `parent`'s
  extension `extension` is loaded (`Base.get_extension(parent, extension)`), else
  `throw(missing_error())`. A function calls it first, before any work that would end in
  the extension.
- `require_extension(missing_error, f, args, parent, extension)`: the fallback method of
  a function `f` whose methods live in the extension, called with the arguments
  `args` it received. Without the extension it throws `missing_error()`. With the
  extension loaded the arguments fit none of its methods, so it throws the
  `MethodError(f, args)` the call would have raised, at the caller's world age
  (`Base.tls_world_age()`), so that a call compiled before the extension loaded
  keeps Julia's "method too new" hint.

```julia
_fit(args...; kwargs...) = require_extension(_fit, args, @__MODULE__, :MyPackageEstimationExt) do
    ArgumentError("_fit needs `using VisualGeometryToolkitEstimation` (its package extension is not loaded)")
end
```
"""
require_extension(missing_error, parent::Module, extension::Symbol) =
    Base.get_extension(parent, extension) === nothing ? throw(missing_error()) : nothing
function require_extension(missing_error, f, args::Tuple, parent::Module, extension::Symbol)
    Base.get_extension(parent, extension) === nothing && throw(missing_error())
    throw(MethodError(f, args, Base.tls_world_age()))
end
