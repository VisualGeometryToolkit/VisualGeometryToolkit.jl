# =============================================================================
# Geometry Utilities
# =============================================================================

# ustrip for HyperRectangle (Rect)
"""
    ustrip(u, r::HyperRectangle)

Remove units from HyperRectangle, expressing magnitudes in unit `u`.
"""
FlexUnits.ustrip(u::Units, r::HyperRectangle) = HyperRectangle(ustrip.(u, r.origin), ustrip.(u, r.widths))

# Size2 computed accessors (replacing the former getproperty override,
# which made every Size2 field access dynamically dispatched)
"""
    aspect_ratio(s::Size2)

Width/height ratio.
"""
aspect_ratio(s::Size2) = s.width / s.height

"""
    area(s::Size2)

Width × height (unit-carrying if the fields carry units).
"""
area(s::Size2) = s.width * s.height

# Round method for HyperRectangle (which is what Rect actually is)
"""
    Base.round(::Type{T}, r::HyperRectangle)

Round HyperRectangle to a specific type.
"""
function Base.round(::Type{T}, r::HyperRectangle{N, S}) where {T <: Real, S, N}
    # Use broadcasting to round each component of the origin and widths
    rounded_origin = round.(T, r.origin)
    rounded_widths = round.(T, r.widths)

    return HyperRectangle{N, T}(rounded_origin, rounded_widths)
end

"""
    centroid(r::HyperRectangle)

Get the centroid (geometric center) of a HyperRectangle (Rect/Rect2).

Note: This returns the midpoint, not the origin. For the corner/origin,
use `r.origin` or `GeometryBasics.origin(r)`.

# Examples
```julia
rect = Rect2(0.5, 0.5, 1920.0, 1200.0)
c = centroid(rect)  # [960.5, 600.5]
```
"""
centroid(r::HyperRectangle) = r.origin .+ r.widths ./ 2

# Size2 constructor from Rect using widths field
"""
    Size2(r::Rect{N,T}) where {N,T}

Create a Size2 from a Rect using its widths field.
"""
Size2(r::Rect) = Size2(width = r.widths[1], height = r.widths[2])

"""
    Rect(s::Size2{T}) where T

Create a Rect from a Size2 with origin at (0,0) and widths filled from the size dimensions.
"""
Rect(s::Size2{T}) where {T} = Rect(Point2{T}(zero(T), zero(T)), Vec2{T}(s.width, s.height))

# NOTE: pixel-origin conventions are coordinate FRAMES, not values — the
# `ImageOrigin{S}` singleton value API that lived here (COLMAP_ORIGIN,
# JULIA_ORIGIN, `Translation(::ImageOrigin, ::ImageOrigin)`) was removed in
# 0.7. Origin conversion is now the labeled frame map
# `EuclideanMap(PixelFrame{:colmap}, PixelFrame{:julia})` — see
# transforms/coord_maps.jl and the `PixelFrame` docstring.

# Generic StaticVector constructors that work with both Point2 and Vec2

# =============================================================================
# Canonical Basis Vectors
# =============================================================================

"""
    basis_vector(::Type{SVector{N,T}}, i::Int) where {N,T}

Generate the i-th canonical basis vector in N dimensions.

The canonical basis for ℝⁿ consists of vectors with 1 in the i-th position and 0 elsewhere.
This function is type-stable and returns a StaticVector for compile-time size optimization.

# Arguments
- `SVector{N,T}`: Type of vector to generate (N dimensions, element type T)
- `i::Int`: Index of the basis vector (1 ≤ i ≤ N)

# Returns
The i-th canonical basis vector as an `SVector{N,T}`

# Examples
```julia
# 3D basis vectors
e1 = basis_vector(SVector{3,Float64}, 1)  # [1.0, 0.0, 0.0]
e2 = basis_vector(SVector{3,Float64}, 2)  # [0.0, 1.0, 0.0]
e3 = basis_vector(SVector{3,Float64}, 3)  # [0.0, 0.0, 1.0]

# 2D integer basis
ex = basis_vector(SVector{2,Int}, 1)      # [1, 0]

# 4D with Float32
e4 = basis_vector(SVector{4,Float32}, 4)  # [0.0f0, 0.0f0, 0.0f0, 1.0f0]
```

See also: [`canonical_basis`](@ref)
"""
@inline function basis_vector(::Type{SVector{N, T}}, i::Int) where {N, T}
    @boundscheck 1 ≤ i ≤ N || throw(BoundsError("basis_vector index $i out of range 1:$N"))
    return SVector{N, T}(ntuple(j -> j == i ? one(T) : zero(T), N))
end

"""
    canonical_basis(::Type{SVector{N,T}}) where {N,T}

Generate all N canonical basis vectors for ℝⁿ as a tuple.

Returns a tuple containing all canonical basis vectors e₁, e₂, ..., eₙ where eᵢ
has 1 in the i-th position and 0 elsewhere. This is type-stable and compile-time
optimized using StaticArrays.

# Arguments
- `SVector{N,T}`: Type of vectors to generate (N dimensions, element type T)

# Returns
An N-tuple of `SVector{N,T}` containing all canonical basis vectors

# Examples
```julia
# 3D basis
e1, e2, e3 = canonical_basis(SVector{3,Float64})
# e1 = [1.0, 0.0, 0.0]
# e2 = [0.0, 1.0, 0.0]
# e3 = [0.0, 0.0, 1.0]

# 2D basis with Float32
ex, ey = canonical_basis(SVector{2,Float32})

# Use in transformation
R = RotMatrix(...)  # 3×3 rotation matrix
e1, e2, e3 = canonical_basis(SVector{3,Float64})
x_axis = R * e1  # Transform basis vectors
y_axis = R * e2
z_axis = R * e3
```

See also: `VisualGeometryCore.basis_vector` (unexported)
"""
@inline function canonical_basis(::Type{SVector{N, T}}) where {N, T}
    return ntuple(i -> basis_vector(SVector{N, T}, i), N)
end
