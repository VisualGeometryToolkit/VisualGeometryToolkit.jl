# =============================================================================
# OrientedCamera — what a camera with no position refuses, and its eltype
# =============================================================================
#
# An `OrientedCamera` (cameras.jl) is a `PosedCamera` whose extrinsics are a
# `RotationMap`: it projects, back-projects and unprojects about the origin
# through the map interface, and `pose`, `extrinsics`, `is_in_front` and the
# unit verbs answer for it there. What needs the camera's position refuses it
# here, by name: the plane-coordinate verbs, whose plane z = 0 contains
# the camera's centre (every point of it images onto one line), a rig member
# (its pose is rigid, its frame placed in the rig's), and a non-central grid
# model's pose (its rays start at points in metres, which a rotation cannot
# state the unit of). Included after calibration.jl, whose `RadialCamera` it
# refuses.

const _OrientedPinholeCamera = PosedCamera{<:CameraModel{<:AbstractIntrinsics, Pinhole}, <:RotationMap{3}}

@noinline _refuse_oriented(what) = throw(
    ArgumentError(
        "$(what): the camera is an OrientedCamera, with an orientation and no position; the plane z = 0 of a " *
            "plane-coordinate verb contains its centre, where the whole plane images onto one line — pose " *
            "the camera by an EuclideanMap"
    )
)

HomographyMat(::_OrientedPinholeCamera) = _refuse_oriented("HomographyMat(camera)")
RadialCamera(::PosedCamera{<:CameraModel, <:RotationMap{3}}) = _refuse_oriented("RadialCamera(camera)")
transfer_ellipse(::OrientedCamera, ::Union{Circle, Ellipse}) = _refuse_oriented("transfer_ellipse")
project(::OrientedCamera, ::Union{Circle, Ellipse}) = _refuse_oriented("project(camera, conic)")
project(::OrientedCamera, ::GeometryBasics.HyperRectangle{2}, n::Int = 20) = _refuse_oriented("project(camera, rect)")

_raw_extrinsics(::OrientedCamera) = throw(
    ArgumentError(
        "a rig member is posed rigidly in the rig's frame, and an OrientedCamera has no position there — " *
            "pose the sensor by an EuclideanMap"
    )
)

@noinline _refuse_oriented_noncentral() = throw(
    ArgumentError(
        "PosedCamera: a non-central grid model's rays start at points in metres, and a rotation-only pose " *
            "cannot state the scene's unit — pose it by a physical EuclideanMap, `EuclideanMap(R, zeros(3) .* m)`"
    )
)
PosedCamera(::_NoncentralGrid, ::RotationMap{3}) = _refuse_oriented_noncentral()
PosedCamera(::_NoncentralGrid, ::RotationMap{3, T, R, CameraFrame, WorldFrame}) where {T, R} = _refuse_oriented_noncentral()

# A rotation map in another float type, its representation and tags kept.
convert_eltype(::Type{T}, M::RotationMap{N, S, R, Src, Dst}) where {T <: Real, N, S, R, Src, Dst} =
    relabel(Src, Dst, RotationMap(_rotation_eltype(T, M.R); check = false))
# A plain rigid map in another number type: its rotation's representation and
# its frames kept, the translation converted entry by entry (the isbits plain
# form a kernel or a forward-differentiated Jacobian takes). A unitful map is stripped first
# (`ustrip(u, E)`): its translation holds quantities, whose number type is not
# this conversion's.
convert_eltype(::Type{T}, E::EuclideanMap{N, S, R, V, Src, Dst}) where {T <: Real, N, S, R, V <: StaticVector{N, <:Real}, Src, Dst} =
    relabel(Src, Dst, EuclideanMap(_rotation_eltype(T, E.R), SVector{N, T}(E.t); check = false))
_rotation_eltype(::Type{T}, R::RotMatrix{N}) where {T, N} = RotMatrix{N, T}(R)
_rotation_eltype(::Type{T}, q::Rotations.QuatRotation) where {T} = Rotations.QuatRotation{T}(q)
_rotation_eltype(::Type{T}, a::Angle2d) where {T} = Angle2d{T}(a.theta)

# An oriented camera in another float type: its model and its rotation.
function convert_eltype(::Type{T}, c::OrientedCamera) where {T <: Real}
    model, X = convert_eltype(T, c.model), convert_eltype(T, c.extrinsics)
    return PosedCamera{typeof(model), typeof(X)}(model, X)
end
