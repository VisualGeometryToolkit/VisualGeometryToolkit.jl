# ==============================================================================
# ProjectiveIntrinsics — the principal point alone
# ==============================================================================

"""
    ProjectiveIntrinsics{T <: Real}(center)
    ProjectiveIntrinsics(center)

The intrinsics of a projective ("partially calibrated") camera: the principal
point `center` (`Point2`, COLMAP pixels) and nothing else. Its ray frame is
PROJECTIVE: centred pixels `u − center`, defined only up to the plane
homography that absorbs the focal length. There is no focal-length field, and
none is implied — in particular it is not `f = 1`. Anything that needs a metric
ray has no method and raises a `MethodError` (`focal_length`,
`calibration_matrix`, `unproject`, P3P, absolute pose), except the field of
view, which is refused with an `ArgumentError`: `angle_of_view` and `fov_radius`, and a
finite `max_fov` in `undistorted_rect` (use `max_fov = Inf`).

`CameraModel(ProjectiveIntrinsics(c), lens)` projects, back-projects and
undistorts exactly as `CameraModel(LogicalIntrinsics(K), lens)` with the unit
focal `K = [1 0 cx; 0 1 cy; 0 0 1]` does: both go through the same code, which
reads the pixel matrix at `_K` / `_K_inv`.

`ProjectiveIntrinsics{T}(center)` stores the centre in `T` (a floating-point or AD
eltype; an integer or rational `T` is an `ArgumentError`); `ProjectiveIntrinsics(center)`
takes `T` from `center` (integers become `Float64`). See [`principal_point`](@ref),
[`parameter_names`](@ref) (`(:cx, :cy)`), [`convert_eltype`](@ref) and
[`convert_frame`](@ref).

```jldoctest
julia> using VisualGeometryToolkitCore

julia> ProjectiveIntrinsics(Point2(2120.0, 1416.0))
ProjectiveIntrinsics(cx = 2120.0, cy = 1416.0)
```
"""
struct ProjectiveIntrinsics{T <: Real} <: AbstractIntrinsics
    center::Point2{T}
    function ProjectiveIntrinsics{T}(center::AbstractVector{<:Real}) where {T <: Real}
        # a centre is an observation: floats (and AD numbers, which are `Real`), never exact integers or rationals
        T <: Union{Integer, Rational} && throw(
            ArgumentError("ProjectiveIntrinsics{$T}: the eltype is a floating-point (or AD) type; use ProjectiveIntrinsics(center), which takes float($T)")
        )
        return new{T}(Point2{T}(center[1], center[2]))
    end
end
ProjectiveIntrinsics(center::AbstractVector{T}) where {T <: Real} =
    ProjectiveIntrinsics{float(T)}(center)

"""
    principal_point(intrinsics::ProjectiveIntrinsics{T}) -> Point2{T}
    principal_point(intrinsics::LogicalIntrinsics{T}) -> Point2{T}

The principal point, in COLMAP pixels: a projective camera's `center`, a logical
camera's `(K[1, 3], K[2, 3])` — the one accessor for both kinds of intrinsics.
"""
principal_point(intrinsics::ProjectiveIntrinsics) = intrinsics.center
principal_point(intrinsics::LogicalIntrinsics{T}) where {T} = (K = intrinsics.K; Point2{T}(K[1, 3], K[2, 3]))

# The implicit unit-focal K — an internal detail of the shared project /
# backproject path, never a public accessor (no `calibration_matrix`, no focal).
@inline function _K(intrinsics::ProjectiveIntrinsics{T}) where {T}
    c = intrinsics.center
    return SMatrix{3, 3, T, 9}(one(T), zero(T), zero(T), zero(T), one(T), zero(T), c[1], c[2], one(T))
end
@inline function _K_inv(intrinsics::ProjectiveIntrinsics{T}) where {T}
    c = intrinsics.center
    return SMatrix{3, 3, T, 9}(one(T), zero(T), zero(T), zero(T), one(T), zero(T), -c[1], -c[2], one(T))
end

# The intrinsics kinds with a pixel matrix at `_K` / `_K_inv`, the ones the shared centred-pixel maps take.
const _PixelMatrixIntrinsics = Union{LogicalIntrinsics, ProjectiveIntrinsics}

Base.:(==)(a::ProjectiveIntrinsics, b::ProjectiveIntrinsics) = a.center == b.center
Base.isequal(a::ProjectiveIntrinsics, b::ProjectiveIntrinsics) = isequal(a.center, b.center)
Base.hash(a::ProjectiveIntrinsics, h::UInt) = hash(a.center, hash(:ProjectiveIntrinsics, h))

function Base.show(io::IO, intrinsics::ProjectiveIntrinsics)
    c = intrinsics.center
    return print(io, "ProjectiveIntrinsics(cx = ", c[1], ", cy = ", c[2], ")")
end
