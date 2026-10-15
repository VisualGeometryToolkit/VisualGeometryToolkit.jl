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
# saturation_bounds — the pixel value range a measurement may trust
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
