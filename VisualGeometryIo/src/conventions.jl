# =============================================================================
# Lens conventions — OpenCV's and Kalibr's models as VGC lenses, exactly
# =============================================================================
#
# A convention is an import/export adapter: it maps a foreign model and its
# coefficients to the VGC lens computing the SAME FUNCTION — not to a
# re-ordering of the foreign coefficients — and back, and it refuses a lens or
# a model it cannot represent exactly. COLMAP's models are in assemble.jl,
# MATLAB's in external_calibration.jl.
#
# OpenCV (calib3d `projectPoints`, calibration_base.cpp): the distortion vector
# (k1, k2, p1, p2[, k3[, k4, k5, k6[, s1, s2, s3, s4[, τx, τy]]]]) computes, on
# the perspective point x with r² = |x|²,
#
#   x·(1 + k1r² + k2r⁴ + k3r⁶)/(1 + k4r² + k5r⁴ + k6r⁶) + tangential(p1, p2) + (s1r² + s2r⁴, s3r² + s4r⁴),
#
# then the tilted-sensor projective map of (τx, τy) — every term at the same
# point, the tilt after: `Lens(Perspective(), Chain(Additive(Radial(…), Tangential(…), ThinPrism(…)), Tilt(…)))`.
# The vector's LENGTH is the model (OpenCV's flags), so the lens carries the
# terms that length evaluates, zeros included; `cv::fisheye`'s (k1…k4) is
# `KannalaBrandt`.

const _OPENCV_LENGTHS = (4, 5, 8, 12, 14)

"""
    opencv_lens(c::AbstractVector) -> Lens

The lens computing OpenCV's `projectPoints` distortion for the coefficient
vector `c` = `(k1, k2, p1, p2[, k3[, k4, k5, k6[, s1, s2, s3, s4[, τx, τy]]]])`
— 4, 5, 8, 12 or 14 entries, the terms that length evaluates. Inverse of
[`opencv_coefficients`](@ref).
"""
function opencv_lens(c::AbstractVector{<:Real})
    n = length(c)
    n in _OPENCV_LENGTHS || throw(ArgumentError("an OpenCV distortion vector has 4, 5, 8, 12 or 14 entries; got $n"))
    v = float.(c)
    k = n == 4 ? (v[1], v[2]) : (v[1], v[2], v[5])
    d = n >= 8 ? (v[6], v[7], v[8]) : ()
    terms = (Radial(k, d), Tangential(v[3], v[4]))
    n >= 12 && (terms = (terms..., ThinPrism((v[9], v[10]), (v[11], v[12]))))
    add = Additive(terms...)
    return n == 14 ? Lens(Perspective(), Chain(add, Tilt(v[13], v[14]))) : Lens(Perspective(), add)
end

"""
    opencv_coefficients(lens) -> Vector{Float64}

OpenCV's distortion vector for a lens it computes: a perspective lens whose
distortion is OpenCV's — a radial factor of at most three numerator and three
denominator coefficients, tangential terms, a thin prism of at most two
coefficients per axis, then a tilt — written as the shortest vector (4, 5, 8,
12 or 14 entries) holding all of it. Any other lens is refused.
"""
opencv_coefficients(l::Lens{Perspective, Nothing}) = zeros(4)
opencv_coefficients(l::Lens{Perspective}) = _opencv_vector(_opencv_parts(l.distortion)...)
opencv_coefficients(l::AbstractLensModel) = _not_opencv(l)
_not_opencv(l) = throw(ArgumentError("OpenCV's distortion model does not compute the lens $(typeof(l))"))

# (radial, tangential, thin prism, tilt) of an OpenCV-shaped distortion; `nothing` where absent.
_opencv_parts(r::Radial{Even}) = (r, nothing, nothing, nothing)
_opencv_parts(t::Tangential) = (nothing, t, nothing, nothing)
_opencv_parts(t::Tilt) = (nothing, nothing, nothing, t)
_opencv_parts(c::Chain{<:Tuple{Additive, Tilt}}) = ((r, t, s, _) = _opencv_parts(c.stages[1]); (r, t, s, c.stages[2]))
_opencv_parts(c::Chain{<:Tuple{Radial{Even}, Tilt}}) = (c.stages[1], nothing, nothing, c.stages[2])
function _opencv_parts(a::Additive)
    r, t, s = nothing, nothing, nothing
    for (i, term) in enumerate(a.terms)
        if term isa Radial{Even} && i == 1
            r = term
        elseif term isa Tangential && t === nothing
            t = term
        elseif term isa ThinPrism && s === nothing
            s = term
        else
            throw(ArgumentError("OpenCV's distortion model has no term $(nameof(typeof(term))) there"))
        end
    end
    return r, t, s, nothing
end
_opencv_parts(d) = throw(ArgumentError("OpenCV's distortion model does not compute a $(nameof(typeof(d))) distortion"))

function _opencv_vector(r, t, s, tilt)
    k = r === nothing ? () : r.k
    d = r === nothing ? () : r.d
    (length(k) <= 3 && length(d) <= 3) || throw(ArgumentError("OpenCV's radial factor has at most 3 + 3 coefficients; got $(length(k)) + $(length(d))"))
    (s === nothing || length(s.sx) <= 2) || throw(ArgumentError("OpenCV's thin prism has at most 2 coefficients per axis; got $(length(s.sx))"))
    pad(x, n) = Float64[i <= length(x) ? x[i] : 0.0 for i in 1:n]
    kk, dd = pad(k, 3), pad(d, 3)
    p = t === nothing ? (0.0, 0.0) : (t.p1, t.p2)
    sx, sy = s === nothing ? (zeros(2), zeros(2)) : (pad(s.sx, 2), pad(s.sy, 2))
    v = Float64[kk[1], kk[2], p[1], p[2], kk[3], dd..., sx..., sy..., (tilt === nothing ? (0.0, 0.0) : (tilt.tx, tilt.ty))...]
    n = tilt !== nothing ? 14 : s !== nothing ? 12 : !isempty(d) ? 8 : length(k) == 3 ? 5 : 4
    return v[1:n]
end

# -----------------------------------------------------------------------------
# Kalibr (camchain YAML): camera_model × distortion_model
# -----------------------------------------------------------------------------
#
#   pinhole [fu, fv, pu, pv] + radtan [k1, k2, r1, r2]   OpenCV's 4-coefficient model
#                            + equidistant [k1…k4]       KannalaBrandt
#                            + fov [w]                   FieldOfView(w)
#                            + none []                   Pinhole()
#   omni [ξ, fu, fv, pu, pv] + radtan / none             Lens(Unified(ξ), Additive(Radial((k1, k2)), Tangential(r1, r2)))
#   ds   [ξ, α, fu, fv, pu, pv] + none                   Lens(DoubleSphere(ξ, α))
#   eucm [α, β, fu, fv, pu, pv] + none                   Lens(ExtendedUnified(α, β))
#
# Kalibr's distortions act on the projection's plane point, so `equidistant`
# and `fov` exist only behind the pinhole; any other pairing is refused.

"""
    kalibr_lens(camera_model, intrinsics, distortion_model, distortion_coeffs) -> (fx, fy, cx, cy, lens)

A Kalibr camchain camera (`camera_model` ∈ `pinhole`, `omni`, `ds`, `eucm`;
`distortion_model` ∈ `radtan`, `equidistant`, `fov`, `none`) as its focal
lengths, principal point and the VGC lens computing Kalibr's projection.
"""
function kalibr_lens(camera_model::AbstractString, intrinsics::AbstractVector{<:Real}, distortion_model::AbstractString, coeffs::AbstractVector{<:Real})
    i, c = float.(intrinsics), float.(coeffs)
    want(n, what, v) = length(v) == n || throw(ArgumentError("Kalibr $(what) has $n parameters; got $(length(v))"))
    proj, f = if camera_model == "pinhole"
        want(4, "pinhole", i); Perspective(), i[1:4]
    elseif camera_model == "omni"
        want(5, "omni", i); Unified(i[1]), i[2:5]
    elseif camera_model == "ds"
        want(6, "ds", i); DoubleSphere(i[1], i[2]), i[3:6]
    elseif camera_model == "eucm"
        want(6, "eucm", i); ExtendedUnified(i[1], i[2]), i[3:6]
    else
        throw(ArgumentError("unknown Kalibr camera_model $(repr(camera_model)); known: pinhole, omni, ds, eucm"))
    end
    lens = if distortion_model == "none"
        want(0, "none", c); Lens(proj)
    elseif distortion_model == "radtan" && camera_model in ("pinhole", "omni")
        want(4, "radtan", c); Lens(proj, Additive(Radial((c[1], c[2])), Tangential(c[3], c[4])))
    elseif distortion_model == "equidistant" && camera_model == "pinhole"
        want(4, "equidistant", c); Lens(Equidistant(), Radial(Tuple(c)))
    elseif distortion_model == "fov" && camera_model == "pinhole"
        want(1, "fov", c); Lens(Perspective(), FieldOfView(c[1]))
    else
        throw(ArgumentError("Kalibr $(camera_model)-$(distortion_model) is not a model Kalibr defines"))
    end
    return f[1], f[2], f[3], f[4], lens
end

# -----------------------------------------------------------------------------
# ROS PoseWithCovariance: the pose covariance with its blocks swapped
# -----------------------------------------------------------------------------
#
# Ours (VisualGeometryCore conventions/persisted-geometry.md): 6×6 over
# [δω; δt], rotation first, the rotation perturbed on the left in the pose's
# destination frame, the translation additive in the same frame. ROS's
# geometry_msgs/PoseWithCovariance (common_msgs/geometry_msgs/msg/
# PoseWithCovariance.msg:5-9): row-major 6×6 over (x, y, z, rotation about X,
# Y, Z), fixed-axis rotations in the parent (header) frame — the same chart,
# translation first. The conversion is the block swap P = [0 I; I 0].

const _ROS_SWAP = SMatrix{6, 6, Float64, 36}(ntuple(k -> Float64(mod1(k, 6) == mod1((k - 1) ÷ 6 + 4, 6)), 36))

"""
    ros_pose_covariance(Σ) -> SMatrix{6, 6, Float64}

A pose covariance in the stack's order ([δω; δt], rotation first) as ROS
`PoseWithCovariance.covariance` orders it (x, y, z, rot X, rot Y, rot Z):
`P·Σ·Pᵀ` with `P = [0 I; I 0]`. Flatten row-major (`vec(permutedims(Σ_ros))`)
for the message's `float64[36]`. Inverse of [`pose_covariance_from_ros`](@ref).
"""
ros_pose_covariance(Σ::AbstractMatrix{<:Real}) = _ROS_SWAP * SMatrix{6, 6, Float64, 36}(Σ) * _ROS_SWAP'

"""
    pose_covariance_from_ros(Σ_ros) -> CovMat{6}

A ROS `PoseWithCovariance` covariance (x, y, z, rot X, rot Y, rot Z; a 6×6, or
the message's row-major `float64[36]`) in the stack's order, rotation first —
the covariance an `Uncertain` pose carries. Inverse of
[`ros_pose_covariance`](@ref).
"""
pose_covariance_from_ros(Σ::AbstractMatrix{<:Real}) = CovMat(_ROS_SWAP * SMatrix{6, 6, Float64, 36}(Σ) * _ROS_SWAP')
pose_covariance_from_ros(v::AbstractVector{<:Real}) =
    (length(v) == 36 || throw(ArgumentError("a ROS pose covariance has 36 entries (6×6 row-major), got $(length(v))")); pose_covariance_from_ros(permutedims(reshape(collect(Float64, v), 6, 6))))
