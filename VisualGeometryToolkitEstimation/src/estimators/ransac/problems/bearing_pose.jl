# =============================================================================
# BearingPoseProblem — absolute pose from 3D points and the RAYS observing them
# =============================================================================
#
# The absolute pose on unit BEARINGS (VGC conventions/bearings.md), for a
# camera the stack holds no `CameraModel` of (a grid or non-central lens):
# the caller back-projects and hands over `X => f`, world points and the
# camera-frame unit bearings observing them, optionally with each ray's origin
# (a GENERALIZED, non-central camera) and its back-projection Jacobian (pixel
# whitening). The `MinimalSolveProblem{Extrinsics3}` instance whose context is
# a `BearingPoseContext` (AbsolutePoseProblem's is a `PoseContext`). Restored
# 2026-10-29 from the bearing form of `refine_absolute_pose` (hoisted from
# RadiometricCamera 2026-10-27) as a problem of the refine interface: its rows
# are that form's, unchanged.
#
# THE RESIDUAL is the logarithm map of S² at the observed bearing f, of the
# predicted one p = normalize(R·X + t − o):
#
#     r = θ · Bᵀp / ‖Bᵀp‖,   θ = atan(‖Bᵀp‖, fᵀp) = ∠(f, p),
#
# B = [b₁ b₂] an orthonormal basis of f's tangent plane (VGC's
# `s2_tangent_basis`). It lives in the tangent plane at the observation, its
# norm is EXACTLY the angle between the two rays (radians), and it is smooth
# for every angle below π — unlike Bᵀp alone (norm sin θ), which is zero for
# the antipodal ray and so would score a point seen behind the camera as a
# perfect fit. Near θ = 0 the series θ/sin θ… is taken in ‖Bᵀp‖² (no square
# root is differentiated at the observation).
#
# PIXEL WHITENING. With the bearings' back-projection Jacobians as an `M`
# column (`backproject_with_jacobian(camera, u) -> (f, M)`, tangent:
# fᵀM = 0, checked here per correspondence), each residual is mapped back to
# the pixel it came from, r_px = (BᵀM)⁻¹·r: a first-order pixel residual for
# any lens, so the scoring and every tolerance are in pixels.
#
# GENERALIZED (non-central) cameras: an `origins` column gives each ray's
# origin o in the camera frame; the predicted ray is then from o to the point.
# Without the column every origin is the camera centre. The pixel whitening
# then holds o fixed: (BᵀM)⁻¹ maps the ray DIRECTION's error to pixels and
# ignores ∂o/∂u, so for a non-central camera the "pixel" residual is first
# order in the direction only.
#
# REFINE ONLY for a non-central camera: the stack has no generalized minimal
# solver (gP3P), so `solve` refuses a problem with origins; a central one
# solves by VGC's P3P on its bearings.
#
# THE CHART is AbsolutePoseProblem's, VGC's `LeftRigidManifold6` on the
# extrinsics, (R, t) ← (exp(δω)·R, exp(δω)·t + δt); the Jacobian is VGC's
# default (ForwardDiff through the eltype-generic rows).
# =============================================================================

"""
    BearingPoseContext

The data of a [`BearingPoseProblem`](@ref) beside its correspondences: per
correspondence the ray origin in the camera frame (zero for a central camera),
the observation's tangent basis, the whitening (`I₂`, or `(BᵀM)⁻¹` into
pixels) and the caller's fixed √weight.
"""
struct BearingPoseContext
    origins::Vector{SVector{3, Float64}}
    B::Vector{SMatrix{3, 2, Float64, 6}}
    W::Vector{SMatrix{2, 2, Float64, 4}}
    sw::Vector{Float64}
    central::Bool        # no `origins` column: every ray from the camera centre
    pixels::Bool         # an `M` column: the residuals are pixels, else radians
end

"""
    BearingPoseProblem(cs; sampler = nothing)

The absolute-pose problem on unit BEARINGS: `cs` pairs world points with the
camera-frame unit bearings observing them, `X => f` (a `Csponds`, a
`StructArray` or a Pair vector), for a camera the caller back-projects itself.
The model is the camera extrinsics (world → camera, `x_c = R·X + t`, the
`Extrinsics3` of [`AbsolutePoseProblem`](@ref)); the residual is the tangent-
plane logarithm map of the sphere at `f` of the predicted bearing
`normalize(R·X + t − o)`, its norm the angle between the rays.

Optional attribute COLUMNS of `cs`:
- `origins` — each ray's origin in the camera frame: a GENERALIZED
  (non-central) camera. Such a problem is refine-only: there is no
  generalized minimal solver, so `solve` refuses it.
- `M` — each bearing's back-projection Jacobian (3 × 2, from
  `backproject_with_jacobian(camera, u)`), tangent (`fᵀM = 0`, checked): the
  residuals are whitened into PIXELS, `(BᵀM)⁻¹·r`; without it they are radians.
- `weights` — fixed per-correspondence weights (finite, non-negative) on the
  refit rows; the scoring residual is unweighted.

A bearing of norm ≠ 1 (a z = 1 chart point) is an `ArgumentError`, as is a
non-finite point, origin or bearing. The public estimator is the bearing form
of [`refine_absolute_pose`](@ref).
"""
const BearingPoseProblem = MinimalSolveProblem{
    Extrinsics3, 3, Float64, <:Any, <:Any, <:Any, <:Any, <:VgtkCore.MinimalSolver, <:BearingPoseContext,
}

function BearingPoseProblem(
        cs::AbstractVector{<:Pair{<:StaticVector{3, <:Real}, <:StaticVector{3, <:Real}}};
        sampler::Union{Nothing, AbstractSampler} = nothing
    )
    smplr = _problem_sampler(cs, 3, sampler)
    n = length(cs)
    Xs = SVector{3, Float64}[SVector{3, Float64}(X) for X in sources(cs)]
    fs = SVector{3, Float64}[SVector{3, Float64}(f) for f in targets(cs)]
    os_col = VgtkCore.attribute_column(cs, :origins)
    os = os_col === nothing ? fill(zero(SVector{3, Float64}), n) : SVector{3, Float64}[SVector{3, Float64}(o) for o in os_col]
    M_col = VgtkCore.attribute_column(cs, :M)
    w_col = VgtkCore.weights_column(cs)
    sw = w_col === nothing ? ones(n) : sqrt.(Float64.(VgtkCore.check_weights(w_col)))
    for i in 1:n
        (all(isfinite, Xs[i]) && all(isfinite, os[i])) || throw(ArgumentError("BearingPoseProblem: point or origin $i is not finite"))
        all(isfinite, fs[i]) || throw(ArgumentError("BearingPoseProblem: bearing $i is not finite (a pixel the camera cannot back-project)"))
        abs(norm(fs[i]) - 1) <= _UNIT_BEARING_TOL || throw(
            ArgumentError(
                "BearingPoseProblem takes UNIT bearings; bearing $i has norm $(norm(fs[i])). A z = 1 chart point " *
                    "is not a bearing (its norm is 1 only on the optical axis, and a ray at 90° has none): " *
                    "back-project with the camera, `backproject_with_jacobian(camera, u)`."
            )
        )
    end
    Bs = map(fs) do f
        b1, b2 = s2_tangent_basis(f)
        SMatrix{3, 2, Float64, 6}(b1[1], b1[2], b1[3], b2[1], b2[2], b2[3])
    end
    Ws = M_col === nothing ? fill(SMatrix{2, 2, Float64, 4}(I), n) : [_pixel_whitening(fs[i], Bs[i], M_col[i], i) for i in 1:n]
    ctx = BearingPoseContext(os, Bs, Ws, sw, os_col === nothing, M_col !== nothing)
    data = VgtkCore.Csponds(Xs, fs)
    return MinimalSolveProblem{Extrinsics3, 3}(data, nothing, data, nothing, VgtkCore.P3P(), ctx, smplr)
end

# The structural unit-norm margin of a bearing (as `EssentialProblem`'s): a
# z = 1 chart point has norm ≥ 1, ≈ 1.0185 already on a narrow field.
const _UNIT_BEARING_TOL = 1.0e-6

# (BᵀM)⁻¹: the tangent residual at f mapped to the pixel it came from. `M`
# must be tangent at f (bearings.md: a swapped or untangent Jacobian moves the
# residual little and its descent direction a lot), and invertible on it.
function _pixel_whitening(f, B, M, i)
    Ms = SMatrix{3, 2, Float64, 6}(M)
    all(isfinite, Ms) || throw(ArgumentError("BearingPoseProblem: M[$i] is not finite"))
    for k in 1:2
        abs(dot(f, Ms[:, k])) <= _UNIT_BEARING_TOL * norm(Ms[:, k]) || throw(
            ArgumentError(
                "BearingPoseProblem: M[$i] is not tangent to its bearing (fᵀM ≠ 0): it is not this bearing's " *
                    "back-projection Jacobian (`backproject_with_jacobian(camera, u)` returns the pair)"
            )
        )
    end
    A = B' * Ms
    d = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    (isfinite(d) && !iszero(d)) || throw(ArgumentError("BearingPoseProblem: M[$i] is singular on the tangent plane"))
    return SMatrix{2, 2, Float64, 4}(inv(A))
end

# log_f(p) in the basis B: the tangent vector at f pointing to p, of norm ∠(f, p).
@inline function _bearing_log(f, B, p)
    s = B' * p
    c = dot(f, p)
    n² = s[1] * s[1] + s[2] * s[2]
    # θ / ‖s‖ = atan(n, c)/n; for n ≪ c its series (1 − n²/3c²)/c, exact to
    # (n/c)⁴/5 ≤ 2e-17 below the switch, and differentiable at n = 0.
    if c > 0 && n² < 1.0e-8 * c * c
        return s * ((1 - n² / (3 * c * c)) / c)
    end
    n = sqrt(n²)
    iszero(n) && return SVector{2}(oftype(c, π), zero(c))       # the antipode: any direction, angle π
    return s * (atan(n, c) / n)
end

# Point i's whitened tangent residual at the pose (eltype-generic: the chart
# LM differentiates it), without the caller's weight.
@inline function _bearing_residual(p::BearingPoseProblem, R, t, i::Int)
    ctx = p.context
    @inbounds begin
        X = sources(p.cs)[i]; f = targets(p.cs)[i]
        q = normalize(R * X + t - ctx.origins[i])
        return ctx.W[i] * _bearing_log(f, ctx.B[i], q)
    end
end

# The scoring residual: the norm of the whitened tangent residual (radians, or
# pixels with an `M` column), the caller's weight not applied.
@inline whitened_residual(p::BearingPoseProblem, pose::PlainEuclideanMap{3, Float64}, i::Int) =
    norm(_bearing_residual(p, rotation(pose), translation(pose), i))

# The unit of the class's file `threshold` (`[pipelines.bearing_pose]`): the
# residual's BOUND — π (the largest angle between two rays) times the largest
# whitening gain — so the file's 1.0 admits every correspondence to the
# consensus set and the default refine is least squares over all of them.
_threshold_unit(p::BearingPoseProblem) = π * maximum(W -> opnorm(W), p.context.W)

# P3P on the bearings for a central camera; nothing solves a generalized one.
function solve(p::BearingPoseProblem, idx::AbstractVector{Int})
    p.context.central || throw(ArgumentError(
        "BearingPoseProblem with ray origins (a non-central camera) has no minimal solver: it is refine-only " *
            "(`refine_absolute_pose(pose, cs)` from a guess)"))
    return solve(p.solver, p.cs_solve, idx)
end

# Degenerate samples: coincident or collinear world points, coplanar rays
# (AbsolutePoseProblem's gate on its back-projected rays).
function test_sample(p::BearingPoseProblem, idx::AbstractVector{Int})
    @inbounds begin
        X = (sources(p.cs)[idx[1]], sources(p.cs)[idx[2]], sources(p.cs)[idx[3]])
        f = (targets(p.cs)[idx[1]], targets(p.cs)[idx[2]], targets(p.cs)[idx[3]])
    end
    _are_points_distinct(X; tol_px = _P3P_COINCIDENT_WORLD) || return false
    thin_ratio(X[1], X[2], X[3]) > _P3P_DEGENERACY_TOL || return false
    return _ray_thin_ratio(f[1], f[2], f[3]) > _P3P_DEGENERACY_TOL
end

test_model(::BearingPoseProblem, pose::PlainEuclideanMap{3, Float64}, ::AbstractVector{Int}) = all(isfinite, translation(pose))

# The pose rows are in chart coordinates (as AbsolutePoseProblem's).
_model_directions(::BearingPoseProblem, _θ) = nothing

# The chart: AbsolutePoseProblem's.
chart_dof(::Type{<:BearingPoseProblem}) = 6
_chart(::BearingPoseProblem, pose) = _reanchored(LeftRigidManifold6(rotation(pose), translation(pose)))

# √w-weighted rows in the chart at `δ` (VGC's `residual_rows!`): the LO's √w
# per row times the caller's fixed √weight per correspondence. The Jacobian is
# VGC's default, ForwardDiff through these rows.
function residual_rows!(r, lm::ChartLm{<:LeftRigidManifold6, <:BearingPoseProblem}, δ)
    p = lm.problem; sqrtw = lm.sqrtw; sw = p.context.sw
    pose = retract(lm.ch, δ)
    R = rotation(pose); t = translation(pose)
    @inbounds for i in 1:data_size(p)
        row = 2i - 1
        e = _bearing_residual(p, R, t, i)
        r[row] = sqrtw[row] * sw[i] * e[1]; r[row + 1] = sqrtw[row + 1] * sw[i] * e[2]
    end
    return r
end
