# =============================================================================
# Assembly — typed conversion between wire records and VGC types (Layer 3)
# =============================================================================
#
# Quantity-first: a physical pose holds its translation as lengths in metres
# (VGC's base form); reads attach the file's declared unit through
# `read_length`, writes leave through `write_length` in the target contract's
# unit, and a plain translation is refused where a metric pose is meant.

# A rotation's quaternion, w-x-y-z with w ≥ 0. A `QuatRotation` — every
# rotation a vgio file is read into — is written as it is (the sign made
# canonical by exact negation), so file → value → file is byte-identical and a
# value read from a file round-trips `==`. Any other rotation (a `RotMatrix`
# computed in memory) is converted once, and reads back as the quaternion.
_qvec(R::QuatRotation) = canonical_qvec(SVector{4, Float64}(params(R)))
_qvec(R) = canonical_qvec(SVector{4, Float64}(params(QuatRotation(RotMatrix{3, Float64}(R)))))

# A stored pose as the value the readers return: a rigid map whose rotation IS
# the stored unit quaternion — not renormalized, not converted to a matrix —
# and whose translation is lengths in metres, the static vector VGC's own
# physical constructor stores.
_quat_map(q, t::SVector{3, Quantity{Float64, Dimensions{FixRat32}}}) =
    EuclideanMap{3, Float64, QuatRotation{Float64}, typeof(t)}(
    QuatRotation(Float64(q[1]), Float64(q[2]), Float64(q[3]), Float64(q[4]), false), t,
)

# --- poses ↔ COLMAP images ----------------------------------------------------

"""
    colmap_image(pose::EuclideanMap{3}, image_id, camera_id, name; contract = MM_PX) -> ColmapImage

COLMAP images hold world→camera poses; a detected object→camera pose is the
same direction, with "world" = the object frame. The pose is metric — its
translation lengths, written in the contract's world unit
([`write_length`](@ref)); a plain translation is refused — so an
[`UnknownScale`](@ref) contract, which has no length unit to write it in, is
refused.
"""
function colmap_image(
        pose::EuclideanMap{3}, image_id::Integer, camera_id::Integer,
        name::AbstractString; contract::Union{UnitContract, UnknownScale} = MM_PX
    )
    t = _lengths_json(pose.t, _world_unit(contract), "colmap_image: the pose's translation")
    return ColmapImage(
        Int32(image_id), _qvec(pose.R), SVector{3, Float64}(t),
        Int32(camera_id), String(name)
    )
end

"""
    rigid_transform(im::ColmapImage, contract) -> EuclideanMap{3}

Assemble the world→camera pose of a COLMAP image row under the model's unit
contract: its translation lengths in metres, read in the contract's world unit
([`read_length`](@ref)). An `EuclideanMap` is metric, so under
[`UnknownScale`](@ref) this refuses; the row's `qvec` and unitless `tvec` are
the pose up to scale.
"""
rigid_transform(im::ColmapImage, contract::Union{UnitContract, UnknownScale}) =
    _quat_map(im.qvec, _lengths_from_json(Val(3), im.tvec, _world_unit(contract)))

# --- cameras ↔ COLMAP cameras --------------------------------------------------
#
# Every COLMAP 4.0 camera model is a VGC lens computing COLMAP's own function
# (src/colmap/sensor/models.h): the reader builds the lens of the model's
# structure — its arity is the model's, zeros included — and the writer picks
# the COLMAP model whose structure contains the lens's, padding with zeros.
# Exact or refused: a lens no COLMAP model computes is never approximated.
#
#   SIMPLE_PINHOLE, PINHOLE        Pinhole()
#   SIMPLE_RADIAL, RADIAL          Lens(Perspective(), Radial((k,)) | Radial((k1, k2)))
#   OPENCV                         Lens(Perspective(), Additive(Radial((k1, k2)), Tangential(p1, p2)))
#   FULL_OPENCV                    … Radial((k1, k2, k3), (k4, k5, k6)) …
#   OPENCV_FISHEYE                 KannalaBrandt(k1, k2, k3, k4)
#   SIMPLE_RADIAL_FISHEYE, RADIAL_FISHEYE   KannalaBrandt(k), KannalaBrandt(k1, k2)
#   FOV                            Lens(Perspective(), FieldOfView(ω))
#   THIN_PRISM_FISHEYE             Lens(Equidistant(), Additive(Radial((k1, …, k4)), Tangential(p1, p2), ThinPrism((sx1,), (sy1,))))
#   RAD_TAN_THIN_PRISM_FISHEYE     Lens(Equidistant(), Chain(Radial((k0, …, k5)), Additive(Tangential(p1, p0), ThinPrism((s0, s1), (s2, s3)))))
#                                  (Aria's p0 multiplies r² + 2x² in x: OpenCV's p2)
#   SIMPLE_DIVISION, DIVISION      Division(k)
#   SIMPLE_FISHEYE, FISHEYE        Lens(Equidistant())

"""
    colmap_model(lens) -> Union{Symbol, Nothing}

The COLMAP camera model whose function the lens computes — the one
[`colmap_camera`](@ref) writes — or `nothing` when no COLMAP model computes
it (a double sphere, a lift with a numerator, a thin prism on a perspective
view). Always a two-focal model; reading a single-focal one (`SIMPLE_RADIAL`)
and writing it back gives the two-focal model of the same function (`OPENCV`).
"""
colmap_model(lens::AbstractLensModel) = (c = _colmap_of(lens); c === nothing ? nothing : first(c))

"""
    colmap_params(lens, fx, fy, cx, cy) -> Vector{Float64}

The parameter vector of [`colmap_model`](@ref)`(lens)`, in COLMAP's documented
order for that model.
"""
function colmap_params(lens::AbstractLensModel, fx, fy, cx, cy)
    c = _colmap_of(lens)
    c === nothing && throw(ArgumentError("no COLMAP camera model computes the lens $(typeof(lens))"))
    return Float64[fx, fy, cx, cy, last(c)...]
end

_pad(t::Tuple, n) = ntuple(i -> i <= length(t) ? Float64(t[i]) : 0.0, n)

# (model, its extra parameters in COLMAP's order) — or nothing.
_colmap_of(::AbstractLensModel) = nothing
_colmap_of(::Lens{Perspective, Nothing}) = :PINHOLE => ()
_colmap_of(l::Lens{Perspective, <:Radial{Even}}) = _opencv_colmap(l.distortion, (0.0, 0.0))
_colmap_of(l::Lens{Perspective, <:Tangential}) = _opencv_colmap(Radial(()), (l.distortion.p1, l.distortion.p2))
_colmap_of(l::Lens{Perspective, <:Additive{<:Tuple{Radial{Even}, Tangential}}}) =
    _opencv_colmap(l.distortion.terms[1], (l.distortion.terms[2].p1, l.distortion.terms[2].p2))
function _opencv_colmap(r::Radial, p)
    k, d = r.k, r.d
    isempty(d) && length(k) <= 2 && return :OPENCV => (_pad(k, 2)..., p...)
    (length(k) <= 3 && length(d) <= 3) || return nothing
    kk, dd = _pad(k, 3), _pad(d, 3)
    return :FULL_OPENCV => (kk[1], kk[2], p..., kk[3], dd...)
end
_colmap_of(l::Lens{Perspective, <:FieldOfView}) = :FOV => (l.distortion.omega,)
_colmap_of(::Lens{Equidistant, Nothing}) = :FISHEYE => ()
function _colmap_of(l::Lens{Equidistant, <:Radial{Even}})
    r = l.distortion
    isempty(r.d) || return nothing
    length(r.k) <= 4 && return :OPENCV_FISHEYE => _pad(r.k, 4)
    length(r.k) <= 6 && return :RAD_TAN_THIN_PRISM_FISHEYE => (_pad(r.k, 6)..., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return nothing
end
function _colmap_of(l::Lens{Equidistant, <:Additive{<:Tuple{Radial{Even}, Tangential, Vararg{ThinPrism{1}, N}}}}) where {N}
    r, t = l.distortion.terms[1], l.distortion.terms[2]
    (isempty(r.d) && length(r.k) <= 4 && N <= 1) || return nothing
    sx, sy = N == 1 ? (l.distortion.terms[3].sx[1], l.distortion.terms[3].sy[1]) : (0.0, 0.0)
    k = _pad(r.k, 4)
    return :THIN_PRISM_FISHEYE => (k[1], k[2], t.p1, t.p2, k[3], k[4], sx, sy)
end
function _colmap_of(l::Lens{Equidistant, <:Chain{<:Tuple{Radial{Even}, Additive{<:Tuple{Tangential, ThinPrism}}}}})
    r, a = l.distortion.stages
    t, s = a.terms
    (isempty(r.d) && length(r.k) <= 6 && length(s.sx) <= 2) || return nothing
    sx, sy = _pad(s.sx, 2), _pad(s.sy, 2)
    return :RAD_TAN_THIN_PRISM_FISHEYE => (_pad(r.k, 6)..., t.p2, t.p1, sx..., sy...)
end
_colmap_of(l::Division{1}) = :DIVISION => (l.radial.d[1],)

"""
    colmap_camera(cm::CameraModel, camera_id, width, height) -> ColmapCamera

Serialize a VGC camera as the COLMAP model computing its lens's function
([`colmap_model`](@ref)). A lens no COLMAP model computes, and a `K` with
nonzero skew, are refused rather than approximated.
"""
function colmap_camera(cm::CameraModel, camera_id::Integer, width::Integer, height::Integer)
    model = colmap_model(cm.lens)
    model === nothing && throw(
        ArgumentError("no COLMAP camera model computes the lens $(typeof(cm.lens)) — it is not exported")
    )
    K = cm.intrinsics.K
    # No COLMAP model has a skew parameter; dropping one would write a
    # different camera under the same numbers — refused like a lens COLMAP
    # cannot express.
    iszero(K[1, 2]) || throw(
        ArgumentError("COLMAP camera models have no skew; this camera's K carries skew = $(K[1, 2])")
    )
    return ColmapCamera(Int32(camera_id), model, width, height, colmap_params(cm.lens, K[1, 1], K[2, 2], K[1, 3], K[2, 3]))
end

# Read side: the intrinsics and the lens of each model.
_colmap_intrinsics(::Val{M}, p) where {M} =
    M in (:SIMPLE_PINHOLE, :SIMPLE_RADIAL, :RADIAL, :SIMPLE_RADIAL_FISHEYE, :RADIAL_FISHEYE, :SIMPLE_DIVISION, :SIMPLE_FISHEYE) ?
    (p[1], p[1], p[2], p[3]) : (p[1], p[2], p[3], p[4])

_colmap_lens(::Val{M}, p, ::Type{T}) where {M, T} = _colmap_lens_of(Val(M), T.(p[(_colmap_nfocal(M) + 3):end]))
_colmap_nfocal(M) = M in (:SIMPLE_PINHOLE, :SIMPLE_RADIAL, :RADIAL, :SIMPLE_RADIAL_FISHEYE, :RADIAL_FISHEYE, :SIMPLE_DIVISION, :SIMPLE_FISHEYE) ? 1 : 2
_colmap_lens_of(::Val{:SIMPLE_PINHOLE}, e) = Pinhole()
_colmap_lens_of(::Val{:PINHOLE}, e) = Pinhole()
_colmap_lens_of(::Val{:SIMPLE_RADIAL}, e) = Lens(Perspective(), Radial((e[1],)))
_colmap_lens_of(::Val{:RADIAL}, e) = Lens(Perspective(), Radial((e[1], e[2])))
_colmap_lens_of(::Val{:OPENCV}, e) = Lens(Perspective(), Additive(Radial((e[1], e[2])), Tangential(e[3], e[4])))
_colmap_lens_of(::Val{:FULL_OPENCV}, e) =
    Lens(Perspective(), Additive(Radial((e[1], e[2], e[5]), (e[6], e[7], e[8])), Tangential(e[3], e[4])))
_colmap_lens_of(::Val{:OPENCV_FISHEYE}, e) = KannalaBrandt(e[1], e[2], e[3], e[4])
_colmap_lens_of(::Val{:SIMPLE_RADIAL_FISHEYE}, e) = KannalaBrandt(e[1])
_colmap_lens_of(::Val{:RADIAL_FISHEYE}, e) = KannalaBrandt(e[1], e[2])
_colmap_lens_of(::Val{:FOV}, e) = Lens(Perspective(), FieldOfView(e[1]))
_colmap_lens_of(::Val{:THIN_PRISM_FISHEYE}, e) =
    Lens(Equidistant(), Additive(Radial((e[1], e[2], e[5], e[6])), Tangential(e[3], e[4]), ThinPrism((e[7],), (e[8],))))
_colmap_lens_of(::Val{:RAD_TAN_THIN_PRISM_FISHEYE}, e) = Lens(
    Equidistant(), Chain(Radial(Tuple(e[1:6])), Additive(Tangential(e[8], e[7]), ThinPrism((e[9], e[10]), (e[11], e[12]))))
)
_colmap_lens_of(::Val{:SIMPLE_DIVISION}, e) = Division(e[1])
_colmap_lens_of(::Val{:DIVISION}, e) = Division(e[1])
_colmap_lens_of(::Val{:SIMPLE_FISHEYE}, e) = Lens(Equidistant())
_colmap_lens_of(::Val{:FISHEYE}, e) = Lens(Equidistant())
_colmap_lens_of(::Val{M}, e) where {M} = throw(ArgumentError("no lens assembly for COLMAP model $M"))

"""
    camera_model([T = Float64,] c::ColmapCamera) -> CameraModel

Assemble a VGC camera from a COLMAP row. `T = Float32` gives single-precision
storage end-to-end (GPU pipelines); the wire stays Float64 regardless —
precision is an in-memory concern, not a storage one.
"""
function camera_model(::Type{T}, c::ColmapCamera) where {T <: AbstractFloat}
    fx, fy, cx, cy = _colmap_intrinsics(Val(c.model), c.params)
    K = CameraCalibrationMat(SMatrix{3, 3, T}(fx, 0, 0, 0, fy, 0, cx, cy, 1))
    lens = _colmap_lens(Val(c.model), c.params, T)
    return CameraModel(K; lens, image_origin = :colmap)
end

camera_model(c::ColmapCamera) = camera_model(Float64, c)
