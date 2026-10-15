# =============================================================================
# Coordinate conversions: euclidean ↔ affine (homogeneous w=1) ↔ projective
# =============================================================================
#
# Pure container-preserving utilities — no dependence on VGC-specific types.
# Loaded early so Primitives, transforms, and solvers can all use them.
#
#   to_affine    : v::StaticVector{N}      → similar_type{N+1} appended with 1
#   to_euclidean : v::AbstractVector{N+1}  → SVector{N} via perspective division
#
# THE perspective division of the stack is CoordinateTransformations'
# `PerspectiveMap()` (`pop(v) * inv(v[end])` on a StaticVector: static type
# kept, Dual-safe). Every dehomogenization in VGC calls it — `to_euclidean`,
# `project(Pinhole(), X)`, `undistort_normalized`, the transfer residual — never
# a hand-written `v[1]/v[3]`, which differs from it by an ulp.

"""
    to_affine(v::StaticVector{N,T}) -> StaticVector{N+1,T}

Convert euclidean coordinates to affine coordinates by appending 1 (setting w=1).
Embeds points in the affine hyperplane of projective space.  `vcat` preserves
the input wrapper (Point2 → Point3, Vec2 → Vec3, SVector{2} → SVector{3}).

For a non-static `AbstractVector` fallback, returns `SVector{N+1,T}`.
"""
to_affine(v::StaticVector{N, T}) where {N, T} = vcat(v, one(T))

function to_affine(v::AbstractVector{T}) where {T}
    n = length(v)
    return SVector{n + 1, T}(v..., one(T))
end

"""
    to_euclidean(v::AbstractVector{T}) -> SVector{N-1,T}

Convert homogeneous coordinates to euclidean by perspective division.
Removes the last coordinate after dividing by it.
"""
function to_euclidean(v::AbstractVector{T}) where {T}
    n = length(v)
    n >= 2 || throw(ArgumentError("Vector must have at least 2 elements for euclidean conversion"))
    iszero(v[end]) && throw(ArgumentError("Cannot convert to euclidean: homogeneous coordinate is zero"))
    d = PerspectiveMap()(v)   # THE perspective division (CoordinateTransformations'; see the header)
    return SVector{n - 1, eltype(d)}(Tuple(d))
end

"""
    to_euclidean(v::StaticVector{3}, w_ref::Real) -> Union{SVector{2}, Nothing}

Chirality-guarded perspective division: returns `nothing` unless the
homogeneous coordinate is finite, nonzero, and on the SAME SIDE as `w_ref`
(the w a reference point on the visible side produced — see
[`homogeneous_w`](@ref)). Points beyond a plane's horizon flip w's sign and
would otherwise wrap into view from behind the camera; the throwing one-arg
form cannot express that, so callers hand-rolled this guard (three copies in
BlobBoards, one of which forgot it — the 2026-09 audit's training-mask bug).
"""
function to_euclidean(v::StaticVector{3}, w_ref::Real)
    w = v[3]
    (isfinite(w) && !iszero(w) && sign(w) == sign(w_ref)) || return nothing
    d = SVector{2}(Tuple(PerspectiveMap()(v)))
    all(isfinite, d) || return nothing
    return d
end

# The 2-D cross product (the z-component of [a; 0] × [b; 0]): twice the signed
# area of the triangle (0, a, b). The one spelling, for tuples and static vectors.
@inline _cross2d(a, b) = a[1] * b[2] - a[2] * b[1]
