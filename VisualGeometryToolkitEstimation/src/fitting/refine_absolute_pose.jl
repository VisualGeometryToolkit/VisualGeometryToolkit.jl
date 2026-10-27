# =============================================================================
# refine_absolute_pose — a camera pose from 3D points and their BEARINGS
# =============================================================================
#
# The absolute-pose refinement on unit bearings (VGC conventions/bearings.md):
# given a pose guess (world → camera extrinsics, x_c = R·X + t) and pairs
# `X => f` of world points and the camera-frame unit bearings that observe
# them, minimize the bearings' angular errors. Nothing here ever lifts a ray
# onto the z = 1 plane, so a fisheye's rays at and past 90° — and a ray behind
# the camera — are data like any other.
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
# any lens, so `rms` and every tolerance downstream are in pixels.
#
# GENERALIZED (non-central) cameras: an `origins` column gives each ray's
# origin o in the camera frame (a non-central model's per-pixel line, e.g.
# VGC's `NoncentralGenericModel`); the predicted ray is then from o to the
# point. Without the column every origin is the camera centre.
#
# THE CHART is VGC's `LeftRigidManifold6` on the extrinsics,
# (R, t) ← (exp(δω)·R, exp(δω)·t + δt), the perturbation the stack's pose
# constraint differentiates; the driver is VGC's chart LM (`lmsolve_chart!`,
# NLLSsolver) with its stall postcondition, the Jacobian VGC's default
# (ForwardDiff through the eltype-generic rows).
#
# Hoisted from RadiometricCamera's `fit_pose` / `fit_pose_rays` (a hand-rolled
# Gauss–Newton on the chordal residual normalize(R·X + t − o) − f, right
# perturbation), 2026-10-27: the same model and data, on the stack's chart,
# residual and LM.
# =============================================================================

"""
    AbsolutePoseRefineMetadata

`FitResult` metadata of [`refine_absolute_pose`](@ref): `s²`, the residual
variance `Σ‖rᵢ‖² / ν` (rad², or px² with an `M` column), and `ν = 2n − 6`,
its degrees of freedom. Both forward as properties: `result.s²`, `result.ν`.
"""
struct AbsolutePoseRefineMetadata
    s²::Float64
    ν::Int
end

# The linearization state: the chart anchored at the guess, the data, and per
# correspondence the observation's tangent basis, the whitening (I₂ or
# (BᵀM)⁻¹) and the √weight.
struct _BearingPoseLm{C}
    ch::C
    X::Vector{SVector{3, Float64}}
    f::Vector{SVector{3, Float64}}
    o::Vector{SVector{3, Float64}}
    B::Vector{SMatrix{3, 2, Float64, 6}}
    W::Vector{SMatrix{2, 2, Float64, 4}}
    sw::Vector{Float64}
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

# The predicted bearing of point i at the pose (eltype-generic).
@inline _predicted_bearing(R, t, X, o) = normalize(R * X + t - o)

function residual_rows!(r, lm::_BearingPoseLm, δ)
    pose = retract(lm.ch, δ)
    R = rotation(pose); t = translation(pose)
    @inbounds for i in eachindex(lm.X)
        p = _predicted_bearing(R, t, lm.X[i], lm.o[i])
        e = lm.sw[i] * (lm.W[i] * _bearing_log(lm.f[i], lm.B[i], p))
        r[2i - 1] = e[1]; r[2i] = e[2]
    end
    return r
end

"""
    refine_absolute_pose(pose, cs; max_iter)   # default: configs/default.toml [refine_absolute_pose]
        -> FitResult{<:EuclideanMap{3}, AbsolutePoseRefineMetadata}

Refine the camera extrinsics `pose` (world → camera, `x_c = R·X + t`) against
correspondences `cs` of world points and the unit BEARINGS observing them,
`X => f` (`f` a camera-frame unit 3-vector, from the camera's back-projection —
never a z = 1 point: a bearing of norm ≠ 1 is an `ArgumentError`). Minimizes
`Σ wᵢ‖rᵢ‖²`, `rᵢ` the tangent-plane residual of bearings.md — the logarithm map
of the sphere at `fᵢ` of the predicted bearing `normalize(R·Xᵢ + t − oᵢ)`, whose
norm is the angle between the two rays — over VisualGeometryToolkitCore's
`LeftRigidManifold6` chart by VGC's chart LM. A refine: the guess is required
and the family is kept; rays at and past 90° from the optical axis, and
behind the camera, are handled like any other (fisheye and omnidirectional
cameras): the residual is smooth for every ray error below π.

Optional attribute COLUMNS of `cs` (a `Csponds`, `withattrs(cs; …)`):
- `origins` — each ray's origin in the camera frame: the GENERALIZED
  (non-central) form, `normalize(R·X + t − o) ≈ f`; absent, every ray starts
  at the camera centre.
- `M` — each bearing's back-projection Jacobian (3 × 2, from
  `backproject_with_jacobian(camera, u)`), which must be tangent (`fᵀM = 0`,
  checked): each residual is whitened into PIXELS, `(BᵀM)⁻¹·r`. Absent, the
  residuals are in radians.
- `weights` — fixed per-correspondence weights (finite, non-negative).

`max_iter` caps the LM's accepted iterations. `retcode` is the chart LM's
(`:converged`, `:max_iterations`, `:stalled`, `:solve_failed`, …); `rms` is
`√(Σ wᵢ‖rᵢ‖² / n)`; the metadata carries `s²` and `ν = 2n − 6`. At least 3
correspondences.

# Frames and units
The pose comes back with the input's frame labels and translation form. Plain
inputs give a plain pose; physical ones (a unitful pose, points and origins
that are lengths, all one form) are read in metres at entry and the pose comes
back physical, in metres. The residuals are radians or pixels either way.
"""
function refine_absolute_pose(
        pose::PlainEuclideanMap{3}, cs::AbstractVector{<:Cspond{<:StaticVector{3, <:Real}, <:StaticVector{3, <:Real}}};
        max_iter::Integer = _dflt("refine_absolute_pose", "max_iter")
    )
    n = length(cs)
    n >= 3 || throw(ArgumentError("refine_absolute_pose needs at least 3 correspondences (6 unknowns), got $n"))
    Xs = SVector{3, Float64}[SVector{3, Float64}(X) for X in sources(cs)]
    fs = SVector{3, Float64}[SVector{3, Float64}(f) for f in targets(cs)]
    os_col = VgtkCore.attribute_column(cs, :origins)
    os = os_col === nothing ? fill(zero(SVector{3, Float64}), n) : SVector{3, Float64}[SVector{3, Float64}(o) for o in os_col]
    M_col = VgtkCore.attribute_column(cs, :M)
    w_col = VgtkCore.weights_column(cs)
    sw = w_col === nothing ? ones(n) : sqrt.(Float64.(VgtkCore.check_weights(w_col)))
    for i in 1:n
        (all(isfinite, Xs[i]) && all(isfinite, os[i])) || throw(ArgumentError("refine_absolute_pose: point or origin $i is not finite"))
        all(isfinite, fs[i]) || throw(ArgumentError("refine_absolute_pose: bearing $i is not finite (a pixel the camera cannot back-project)"))
        abs(norm(fs[i]) - 1) <= _UNIT_BEARING_TOL || throw(
            ArgumentError(
                "refine_absolute_pose takes UNIT bearings; bearing $i has norm $(norm(fs[i])). A z = 1 chart point " *
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

    ch = _reanchored(LeftRigidManifold6(SMatrix{3, 3, Float64, 9}(rotation(pose)), SVector{3, Float64}(translation(pose))))
    R0 = rotation(model(ch)); t0 = translation(model(ch))
    for i in 1:n
        iszero(norm(R0 * Xs[i] + t0 - os[i])) &&
            throw(ArgumentError("refine_absolute_pose: point $i coincides with its ray's origin at the guess"))
    end
    lm = _BearingPoseLm(ch, Xs, fs, os, Bs, Ws, sw)
    ews = ManifoldLmWorkspace(2n, 6)
    x, _, conv, iters = lmsolve_chart!(ews, lm; maxit = max_iter, stationarity_tol = _CHART_LM_STATIONARITY_TOL)
    est = retract(ch, x)
    S = sum(abs2, residual_rows!(zeros(2n), lm, x))
    ν = 2n - 6
    out = relabel(_frames(pose)..., EuclideanMap(RotMatrix{3, Float64}(rotation(est)), SVector{3, Float64}(translation(est))))
    return FitResult(out, VgtkCore.lm_retcode(conv), iters, sqrt(S / n), n, AbsolutePoseRefineMetadata(ν > 0 ? S / ν : NaN, ν))
end

# Physical inputs: the pose, the points and the origins in metres once, the
# plain refinement, the pose read back in metres once.
function refine_absolute_pose(
        pose::_PhysicalPose3, cs::AbstractVector{<:Cspond{<:_PhysicalPoint3, <:StaticVector{3, <:Real}}}; kwargs...
    )
    plain = _map_csponds(_plain_metres, identity, cs)
    if plain isa VgtkCore.Csponds && VgtkCore.hasattr(plain, :origins)
        plain = withattrs(plain; origins = map(_plain_metres, plain.origins))
    end
    return _in_metres(refine_absolute_pose(_plain_metres(pose), plain; kwargs...))
end

# The structural unit-norm margin of a bearing (as `EssentialProblem`'s): a
# z = 1 chart point has norm ≥ 1, ≈ 1.0185 already on a narrow field.
const _UNIT_BEARING_TOL = 1.0e-6

# (BᵀM)⁻¹: the tangent residual at f mapped to the pixel it came from. `M`
# must be tangent at f (bearings.md: a swapped or untangent Jacobian moves the
# residual little and its descent direction a lot), and invertible on it.
function _pixel_whitening(f, B, M, i)
    Ms = SMatrix{3, 2, Float64, 6}(M)
    all(isfinite, Ms) || throw(ArgumentError("refine_absolute_pose: M[$i] is not finite"))
    for k in 1:2
        abs(dot(f, Ms[:, k])) <= _UNIT_BEARING_TOL * norm(Ms[:, k]) || throw(
            ArgumentError(
                "refine_absolute_pose: M[$i] is not tangent to its bearing (fᵀM ≠ 0): it is not this bearing's " *
                    "back-projection Jacobian (`backproject_with_jacobian(camera, u)` returns the pair)"
            )
        )
    end
    A = B' * Ms
    d = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    (isfinite(d) && !iszero(d)) || throw(ArgumentError("refine_absolute_pose: M[$i] is singular on the tangent plane"))
    return SMatrix{2, 2, Float64, 4}(inv(A))
end

# The frame labels of a pose, to put back on the refined one.
_frames(::EuclideanMap{3, T, R, V, S, D}) where {T, R, V, S, D} = (S, D)
