# =============================================================================
# Robot-World / Multi-Marker Hand-Eye — staged linear fit
# =============================================================================
#
# The robot-world calibration A_i X_m = Y B_i^m with M markers sharing one Y:
#
#   A_i    = T_WR   rig pose in the reference frame W, per frame i (mocap)
#   B_i^m  = T_CPm  marker-m pose in the camera, per observation (PnP)
#   X_m    = T_RPm  marker m in the rig — M unknowns
#   Y      = T_WC   camera in the reference frame — 1 shared unknown
#
# Unlike `find_hand_eye` (the ratio form A_ij X = X B_ij, which ELIMINATES Y
# and therefore lives in relative-motion space), this estimator keeps Y as a
# parameter and fits all M+1 transforms JOINTLY from the per-(frame, marker)
# pose observations. No RANSAC: the mocap prior makes contamination
# pre-gateable (~0.2% in the motivating dataset).
#
# The estimate is the staged linear solve — Kabsch on pooled relative-motion
# axes → chordal means per marker → one joint linear translation solve. An
# NLLS refinement (FLM on the stacked MRP ⊕ t increment) was implemented and
# REMOVED (2026-08): on the motivating 260527 data the pose-residual MLE
# absorbed an unmodeled systematic (~0.17°, ≈17σ) along a weakly constrained
# direction and extrapolated WORSE than this initializer (3.3/6.3/10.4 vs
# 2.8/4.2/6.7 mm across the range sweep). A measurement-level (reprojection)
# refinement is the sound upgrade path; a pose-residual NLLS is not.
#
# The covariance is the first-order (delta-method) covariance of THIS staged
# estimator: each stage is the stationary point of its own functional (the
# Kabsch alignment, the per-marker chordal means, the translation normal
# equations), so the implicit function theorem on those 6(M + 1) stage
# conditions gives ∂θ̂/∂ε, the estimate's response to the observations'
# perturbations, and cov = (∂θ̂/∂ε) Σ_ε (∂θ̂/∂ε)ᵀ with the per-observation
# noise ŝ²·diag(σ_rot², σ_trans²) of the pose residual below. (The
# Gauss–Newton ŝ²(JᵀJ)⁻¹ of that residual, reported before 2026-09-30, is the
# covariance of the residual's stationary point, which the staged solve is
# not: its Y scattered ~26 % wider than it claimed, NEES median ratio 1.26.)
# The eigen-diagnostics of the residual's JᵀJ are reported alongside: the
# 2026-05 4-frame single-distance calibration pathology (yaw ⊕ t_Y
# unobservable) is exactly what they exist to REPORT, and a rank-deficient
# JᵀJ reports an unbounded (`Inf`) covariance, as `refine_robot_world_hand_eye` does.
#
# The M + 1 transforms live in ONE chart, VGC's `ProductManifold` of
# `RigidExpManifold6` blocks [Y | X₁ … X_M] anchored at the estimate
# (`_robot_world_hand_eye_manifold`): the residual below, the stage conditions, the
# reported covariance and `refine_robot_world_hand_eye` all retract through it.
#
# Frames: inputs are `unlabel`ed at the boundary and the algebra runs on the
# unlabeled `Rigid3` alias (see problems/hand_eye.jl) — honest labels await
# VGC 0.8 view-identity frames (per-instance rig/marker frames).
# =============================================================================

"""
    RobotWorldHandEyeCalibration{K, P}
    RobotWorldHandEyeCalibration(Y, X::AbstractDict{K, <:EuclideanMap{3}})

Result value of [`fit_robot_world_hand_eye`](@ref) and [`refine_robot_world_hand_eye`](@ref) (and
the guess [`refine_robot_world_hand_eye`](@ref) takes): the shared camera-in-reference
transform `Y = T_WC`, the per-marker rig-mount transforms `X[id] = T_RP(id)`
keyed by the caller's marker ids, and `ids`, those keys sorted — the block
order of the covariance on the result's metadata. The constructor derives `ids`
from `X`'s keys and stores the transforms unlabeled, all of one form `P`: plain
(`Rigid3`), or physical — translations in metres — when `Y` and `X` are
(a mix is a `MethodError`).

Robot-world/hand-eye calibration is the literature's name for `AX = YB`
(Zhuang, Roth & Shah 1994; Dornaika & Horaud 1998). Ours is the EYE-TO-HAND
configuration — the camera fixed in the world, the targets on the moving rig —
with frames in the stack's `T_AB` notation: `Y = T_WC`, `X_m = T_RPm`,
`A_i = T_WR` (the rig pose at frame i) and `B_i^m = T_CPm` (marker m in the
camera at frame i). Function names do not encode frames.
"""
struct RobotWorldHandEyeCalibration{K, P <: EuclideanMap{3}}
    Y::P
    X::Dict{K, P}
    ids::Vector{K}
end
function RobotWorldHandEyeCalibration(Y::PlainEuclideanMap{3}, X::AbstractDict{K, <:PlainEuclideanMap{3}}) where {K}
    ids = block_order(X)
    return RobotWorldHandEyeCalibration{K, Rigid3}(
        convert(Rigid3, unlabel(Y)), Dict{K, Rigid3}(id => convert(Rigid3, unlabel(X[id])) for id in ids), ids
    )
end
RobotWorldHandEyeCalibration(Y::_PhysicalPose3, X::AbstractDict{K, <:_PhysicalPose3}) where {K} =
    _in_metres(RobotWorldHandEyeCalibration(_plain_metres(Y), Dict{K, Rigid3}(id => convert(Rigid3, unlabel(_plain_metres(T))) for (id, T) in X)))

# A calibration's two forms, through its poses: the plain one in metres, and a
# plain one read as metres (the boundary of `fit_`/`refine_robot_world_hand_eye`).
_plain_metres(cal::RobotWorldHandEyeCalibration{K, <:_PhysicalPose3}) where {K} =
    RobotWorldHandEyeCalibration{K, Rigid3}(_plain_metres(cal.Y), Dict{K, Rigid3}(id => _plain_metres(T) for (id, T) in cal.X), cal.ids)
function _in_metres(cal::RobotWorldHandEyeCalibration{K, Rigid3}) where {K}
    Y = _in_metres(cal.Y)
    return RobotWorldHandEyeCalibration{K, typeof(Y)}(Y, Dict{K, typeof(Y)}(id => _in_metres(T) for (id, T) in cal.X), cal.ids)
end

# The calibration's chart: VGC's product of `RigidExpManifold6` blocks
# [Y | X_{ids…}] anchored at it — the chart every covariance on the metadata is
# reported in. `Val(B)`, B = M + 1 blocks, keeps the product's type static.
_robot_world_hand_eye_manifold(cal::RobotWorldHandEyeCalibration, ::Val{B}) where {B} =
    ProductManifold(ntuple(b -> RigidExpManifold6(b == 1 ? cal.Y : cal.X[cal.ids[b - 1]]), Val(B))...)
# ... and back: the calibration at a chart's anchor.
function _robot_world_hand_eye_calibration(P::ProductManifold, ids::Vector{K}) where {K}
    poses = model(P)
    return RobotWorldHandEyeCalibration(first(poses), Dict{K, Rigid3}(id => poses[m + 1] for (m, id) in enumerate(ids)))
end

"""
    RobotWorldHandEyeMetadata{K, S}

`FitResult` metadata of [`fit_robot_world_hand_eye`](@ref) (`S = Float64`) and of
[`refine_robot_world_hand_eye`](@ref) (`S = Nothing`: no pose-residual noise
scales; its `retcode` is VGC `lm_retcode`'s) — one type, assembled in one place.
The core carries the `RobotWorldHandEyeCalibration` (`estimate`), the `retcode`, the
`iterations`, the `rms` and the `n` observations (the fit: pose observations,
its `rms` over the weighted residual rows; the refinement: points, its `rms`
the reprojection error per point in pixels). Here:

- `residuals::Dict{K, Vector{Float64}}` — per marker, the residual norm of each
  of its observations in the input order: the fit's weighted 6-residual, the
  refinement's reprojection error (pixels; element by element, each element's
  `X => u` in order)
- `s2::Float64` — the variance factor `RSS/dof` (≈1 for the fit when
  `σ_rot`/`σ_trans` match the data; the pixel variance for the refinement);
  `dof::Int` — the rows minus `6(M + 1)`
- `cov::Matrix{Float64}` — the joint covariance of the estimate in (rad,
  translation-unit) tangent coordinates at it, block order `[Y | X_{ids…}]`,
  each block `(δω, δt)` in the left-perturbation chart `exp(δω)·R, t + δt` (VGC
  `RigidExpManifold6`). The fit's is first order in the observation noise
  `ŝ²·diag(σ_rot², σ_trans²)` through the stages' stationarity conditions (see
  [`fit_robot_world_hand_eye`](@ref)); the refinement's is `s2 · implicit_covariance` of
  its reprojection rows. `Inf` everywhere when `JᵀJ` is numerically rank
  deficient (an unobservable direction — see `min_eigval`).
- `cov_Y::SMatrix{6,6}` — the MARGINAL covariance of `Y` (a sub-block of the
  joint covariance — accounts for the X_m coupling)
- `min_eigval::Float64`, `min_eigvec::Vector{Float64}` — the smallest
  eigenpair of the residual rows' `JᵀJ` (the same chart as `cov`): a near-zero
  `min_eigval` means the calibration geometry leaves the `min_eigvec` direction
  unobservable — REPORTED, never hidden. `NaN` when a row is not finite at the
  estimate (a point the refinement moved out of the lens's domain).
- `σ_rot::S` (rad), `σ_trans::S` — the fit's per-block noise scales of its
  pose residual (given or profiled at the estimate); `nothing` for the
  refinement, whose residual is pixels at unit isotropic noise.
"""
struct RobotWorldHandEyeMetadata{K, S <: Union{Float64, Nothing}}
    residuals::Dict{K, Vector{Float64}}
    s2::Float64
    dof::Int
    cov::Matrix{Float64}
    cov_Y::SMatrix{6, 6, Float64, 36}
    min_eigval::Float64
    min_eigvec::Vector{Float64}
    σ_rot::S
    σ_trans::S
end

# Below this ratio of JᵀJ's smallest to largest eigenvalue the normal matrix is
# numerically singular: an inverse at condition number ≥ 10¹² keeps no more than
# ~4 correct digits in Float64, so the geometry leaves a direction unobservable
# and the covariance is reported unbounded.
const _ROBOT_WORLD_HAND_EYE_RANK_RTOL = 1.0e-12

# THE assembly of a robot-world result — `fit_robot_world_hand_eye`'s and
# `refine_robot_world_hand_eye`'s: the calibration, the residual rows `r` at it (`w` per
# observation, marker by marker in `cal.ids` order, `counts` observations per
# marker) and their Jacobian `J` in the calibration's chart. The smallest
# eigenpair of JᵀJ comes from the singular values of `J` (JᵀJ is never formed
# here); below `_ROBOT_WORLD_HAND_EYE_RANK_RTOL` the covariance is unbounded, otherwise it
# is `covariance(s2)` — the estimator's own, `nothing` when it has none.
function _robot_world_hand_eye_result(
        cal::RobotWorldHandEyeCalibration{K}, r::AbstractVector, J::AbstractMatrix, w::Int, counts, covariance,
        σ_rot::S, σ_trans::S; retcode::Symbol, iterations::Int, rms_rows::Int
    ) where {K, S}
    N = size(J, 2)
    n = length(r) ÷ w
    dof = length(r) - N
    RSS = sum(abs2, r)
    s2 = RSS / dof
    λ_min, v_min, cov = if all(isfinite, J)
        F = svd(J)                                    # singular values descending: JᵀJ's eigenpairs
        λ = F.S .^ 2
        C = λ[end] > _ROBOT_WORLD_HAND_EYE_RANK_RTOL * λ[1] ? covariance(s2) : nothing
        λ[end], Vector(F.V[:, end]), (C !== nothing && all(isfinite, C) ? C : fill(Inf, N, N))
    else
        NaN, fill(NaN, N), fill(Inf, N, N)
    end
    res_norms = split_blocks(point_norms!(Vector{Float64}(undef, n), r, n, w), cal.ids, counts)
    md = RobotWorldHandEyeMetadata{K, S}(
        res_norms, s2, dof, cov, SMatrix{6, 6, Float64, 36}(cov[1:6, 1:6]), λ_min, v_min, σ_rot, σ_trans
    )
    return FitResult(cal, retcode, iterations, sqrt(RSS / rms_rows), n, md)
end

# -----------------------------------------------------------------------------
# Input flattening — the one walk of `fit_robot_world_hand_eye`'s and
# `refine_robot_world_hand_eye`'s keyed markers
# -----------------------------------------------------------------------------

# A rigid transform as static Float64 (R, t), `unlabel`ed at the boundary; a
# non-finite one is named (`what` of marker `id`'s element `j`).
function _finite_pose(T::PlainEuclideanMap{3}, id, j, what)
    a = unlabel(T)
    R = SMatrix{3, 3, Float64, 9}(rotation(a)); t = SVector{3, Float64}(translation(a))
    (all(isfinite, R) && all(isfinite, t)) ||
        throw(ArgumentError("marker $id, element $j: non-finite $what"))
    return R, t
end

# Walk `markers` marker by marker in `ids` order — the block order every
# per-marker output is split back in (VGC `split_blocks`): for element `j` of
# marker `m`, `visit(m, id, j, R_A, t_A, last(c))`, its rig pose `A = first(c)`
# checked finite. `visit` returns how many rows it flattened; a marker that
# flattens none has no `rows` (ArgumentError). Returns the per-marker row counts.
function _flatten_rig_observations(visit, markers, ids, rows::AbstractString)
    counts = zeros(Int, length(ids))
    for (m, id) in enumerate(ids)
        for (j, c) in enumerate(markers[id])
            RA, tA = _finite_pose(first(c), id, j, "rig pose")
            counts[m] += visit(m, id, j, RA, tA, last(c))
        end
        counts[m] > 0 || throw(ArgumentError("marker $id has no $rows"))
    end
    return counts
end

# -----------------------------------------------------------------------------
# Staged linear solve (Kabsch → chordal means → joint linear translations)
# -----------------------------------------------------------------------------
#
# Left-relative motions M_ij = T_i ∘ T_j⁻¹ conjugate by the SHARED Y only:
#   B_i B_j⁻¹ = Y⁻¹ (A_i A_j⁻¹) Y      (X_m cancels within a marker)
# so their angle-scaled rotation axes satisfy  axis(M^B) = R_Yᵀ · axis(M^A),
# pooled over ALL markers and frame pairs — a single Kabsch fit, no sampling.

# Observations are indexed by k (each carries its own rig pose RA[k]); the
# Kabsch pairs are the within-marker observation pairs (k1, k2) whose rig AND
# marker motions both rotate at least `min_rel_angle` — the hand-eye
# construction's predicate (`_rotates_enough`, problems/hand_eye.jl; the angle
# of a relative motion is the same in either composition order). Returned with
# their rig-motion axes, so the covariance differentiates the same stage.
function _robot_world_hand_eye_init(
        RA, tA, obs_marker, RB, tB, by_marker;
        min_rel_angle::Float64    # required: the caller holds the [robot_world_hand_eye] default
    )
    M = length(by_marker)

    # 1. R_Y from pooled relative-motion axis pairs
    as = SVector{3, Float64}[]
    bs = SVector{3, Float64}[]
    pairs = Tuple{Int, Int}[]
    for idxs in by_marker, ii in 1:(length(idxs) - 1)
        k1 = idxs[ii]
        for jj in (ii + 1):length(idxs)
            k2 = idxs[jj]
            ωA = log_so3(SMatrix{3, 3, Float64, 9}(RA[k1] * RA[k2]'))
            ωB = log_so3(SMatrix{3, 3, Float64, 9}(RB[k1] * RB[k2]'))
            (_rotates_enough(ωA, min_rel_angle) && _rotates_enough(ωB, min_rel_angle)) || continue
            push!(as, ωA)
            push!(bs, ωB)
            push!(pairs, (k1, k2))
        end
    end
    length(as) >= 2 || throw(
        ArgumentError(
            "robot-world init: fewer than 2 relative motions above " *
                "$(round(rad2deg(min_rel_angle); digits = 1))° — calibration motion is degenerate"
        )
    )
    R_Y = RotMatrix{3, Float64}(kabsch(as .=> bs)')                   # b = R_Yᵀ a  ⇒  Kabsch gives R_Yᵀ

    # 2. R_X per marker: R_B = R_Yᵀ R_A R_X  ⇒  R_X = R_Aᵀ R_Y R_B, chordal mean
    R_X = [chordal_mean([RotMatrix{3, Float64}(RA[k]' * R_Y * RB[k]) for k in by_marker[m]]) for m in 1:M]

    # 3. All translations jointly:  t_B = R_Yᵀ R_A t_X − R_Yᵀ t_Y + R_Yᵀ t_A
    #    unknown u = [t_Y ; t_X₁ … t_X_M] — exactly optimal given the rotations
    n = length(obs_marker)
    Amat = zeros(3n, 3 * (M + 1))
    rhs = zeros(3n)
    RYt = SMatrix{3, 3, Float64, 9}(R_Y')
    for k in 1:n
        m = obs_marker[k]
        rows = (3k - 2):(3k)
        Amat[rows, 1:3] .= -RYt
        Amat[rows, (3m + 1):(3m + 3)] .= RYt * SMatrix{3, 3, Float64, 9}(RA[k])
        rhs[rows] .= tB[k] - RYt * tA[k]
    end
    u = Amat \ rhs
    t_Y = SVector{3, Float64}(u[1], u[2], u[3])
    t_X = [SVector{3, Float64}(u[3m + 1], u[3m + 2], u[3m + 3]) for m in 1:M]

    return R_Y, t_Y, R_X, t_X, pairs, as
end

function _obs_by_marker(obs_marker, M::Integer)
    by_marker = [Int[] for _ in 1:M]
    for (k, m) in enumerate(obs_marker)
        push!(by_marker[m], k)
    end
    return by_marker
end

# -----------------------------------------------------------------------------
# Joint pose residual — eltype-generic, in the calibration's product chart
# -----------------------------------------------------------------------------
#
# θ = [δY(6) | δX₁(6) … δX_M(6)] in `P` (`_robot_world_hand_eye_manifold`). Used only at
# θ = 0 for the residual value and its ForwardDiff Jacobian (observability
# diagnostics, noise profiling).

function _robot_world_hand_eye_residuals(θ, P::ProductManifold, obs, inv_σr, inv_σt)
    (; RA, tA, obs_marker, RB, tB) = obs
    T = eltype(θ)
    n = length(obs_marker)
    poses = retract(P, θ)                           # (Y, X₁, …, X_M)
    RYt = rotation(first(poses))'
    t_Y = translation(first(poses))

    r = Vector{T}(undef, 6n)
    @inbounds for k in 1:n
        X = poses[obs_marker[k] + 1]
        R_Ai = RA[k]
        # B̂ = Y⁻¹ A_k X_m
        R̂ = RYt * (R_Ai * rotation(X))
        t̂ = RYt * (R_Ai * translation(X) + tA[k] - t_Y)

        R_err = RB[k]' * R̂                                 # measured vs predicted
        ω = log_so3(SMatrix{3, 3, T, 9}(R_err'))           # Dual-safe at identity (RotationVec is NaN there)
        te = tB[k] - t̂

        b = 6 * (k - 1)
        r[b + 1] = inv_σr * ω[1]
        r[b + 2] = inv_σr * ω[2]
        r[b + 3] = inv_σr * ω[3]
        r[b + 4] = inv_σt * te[1]
        r[b + 5] = inv_σt * te[2]
        r[b + 6] = inv_σt * te[3]
    end
    return r
end

# -----------------------------------------------------------------------------
# The staged solve's stationarity conditions — its covariance by the delta method
# -----------------------------------------------------------------------------
#
# F(θ, ε) = 0 at the staged estimate: θ = [δY | δX₁ … δX_M] in the product chart
# about the estimate, ε = [ε₁ … ε_n] the observations' left perturbations
# B_k → (exp(ε_ω)·R_B, t_B + ε_t) (each B_k's own `RigidExpManifold6`). The
# 6(M + 1) rows are
#   Kabsch   the stationarity of `kabsch` (VGC `kabsch_stationarity`) at
#            Q = R_Yᵀ over the rig/marker motion-axis pairs
#   chordal  the stationarity of `chordal_mean` (VGC `chordal_mean_stationarity`)
#            at R_Xm over { R_Akᵀ R_Y R_Bk }
#   transl.  the normal equations of the joint translation least squares.
# By the implicit function theorem ∂θ̂/∂ε = −F_θ⁻¹ F_ε. Eltype-generic: one of
# θ, ε carries the ForwardDiff Duals, the other is zeros.
function _robot_world_hand_eye_stage_conditions(θ, ε, P::ProductManifold, st)
    (; RA, tA, obs_marker, RB, tB, pairs, axes_A, by_marker) = st
    T = promote_type(eltype(θ), eltype(ε))
    M = length(by_marker)
    n = length(obs_marker)

    poses = retract(P, θ)
    R_Y = rotation(first(poses)); t_Y = translation(first(poses))
    Bε = [retract(RigidExpManifold6(RB[k], tB[k]), view(ε, (6k - 5):(6k))) for k in 1:n]
    RBε(k) = rotation(Bε[k])

    F = Vector{T}(undef, 6 * (M + 1))
    F[1:3] .= kabsch_stationarity(
        R_Y', [axes_A[p] => log_so3(SMatrix{3, 3}(RBε(k1) * RBε(k2)')) for (p, (k1, k2)) in enumerate(pairs)]
    )
    nY = zero(SVector{3, T})
    for m in 1:M
        X = poses[m + 1]
        R_X = rotation(X); t_X = translation(X)
        F[(3m + 1):(3m + 3)] .= chordal_mean_stationarity(R_X, [RA[k]' * R_Y * RBε(k) for k in by_marker[m]])
        nX = zero(SVector{3, T})
        for k in by_marker[m]
            R_A = RA[k]
            e = R_Y' * (R_A * t_X + tA[k] - t_Y) - translation(Bε[k])   # row block k of (A u − rhs)
            RYe = R_Y * e
            nY -= RYe
            nX += R_A' * RYe
        end
        F[(3 * (M + 1) + 3m + 1):(3 * (M + 1) + 3m + 3)] .= nX
    end
    F[(3 * (M + 1) + 1):(3 * (M + 1) + 3)] .= nY
    return F
end

# -----------------------------------------------------------------------------
# fit_robot_world_hand_eye
# -----------------------------------------------------------------------------

"""
    fit_robot_world_hand_eye(markers; σ_rot=nothing, σ_trans=nothing, min_rel_angle)
        -> FitResult{RobotWorldHandEyeCalibration{K}, RobotWorldHandEyeMetadata{K, Float64}}

`min_rel_angle` defaults to `configs/default.toml` `[robot_world_hand_eye]`
`min_rel_angle_deg` (5°), converted to radians.

Staged linear robot-world / multi-marker hand-eye calibration
`A_i X_m = Y B_i^m`: one shared `Y = T_WC` and per-marker `X_m = T_RPm`,
fit from per-(frame, marker) pose observations. No RANSAC — with a pose
prior (mocap) contamination is pre-gateable at the call site.

Robot-world/hand-eye calibration is the literature's name for `AX = YB`
(Zhuang, Roth & Shah 1994; Dornaika & Horaud 1998). Ours is the EYE-TO-HAND
configuration — the camera fixed in the world, the targets on the moving rig —
with frames in the stack's `T_AB` notation: `Y = T_WC`, `X_m = T_RPm`,
`A_i = T_WR` (the rig pose at frame i) and `B_i^m = T_CPm` (marker m in the
camera at frame i). Function names do not encode frames.

The estimate is the staged linear solve: Kabsch on pooled relative-motion
axes (`R_Y`), chordal means per marker (`R_Xm`), then one joint linear
least-squares solve for ALL translations — exactly optimal in the
translations given the rotations. A pose-residual NLLS refinement was
deliberately removed: on real single-distance calibration data it absorbed
an unmodeled systematic along a weakly constrained direction and
extrapolated worse than this solution (see the documentation's
"Robot–World Hand–Eye Calibration" page). Measurement-level (reprojection)
refinement is the sound upgrade path.

# Arguments
- `markers::AbstractDict{K, <:AbstractVector{<:Cspond}}` — per marker id, the
  correspondences between rigid transforms `A_i => B_i^m` of every frame it is
  seen in: the rig pose `T_WR` (e.g. mocap) and the marker-in-camera pose
  `T_CPm` (e.g. PnP); any frame labels accepted, `unlabel`ed at the boundary.
  The key is the marker's identity (any type with `isless`); the result's
  `ids` are the keys sorted, and `X` is keyed the same. A frame seen by several
  markers repeats its `A_i` in each marker's vector — nothing is associated
  by index.

# Keywords
- `σ_rot`, `σ_trans` — per-block noise scales (rad, translation units)
  balancing the diagnostic 6-vector residual
  `[Log(R̂ᵀR)/σ_rot; (t−t̂)/σ_trans]`. When `nothing`, profiled as the RMS
  of the corresponding residual block at the estimate (equalizes block
  contributions).
- `min_rel_angle` — a within-marker relative motion enters the Kabsch stage
  only when both its rig and its marker motion rotate at least this angle
  (near-identity motions are uninformative) — the same predicate as
  [`find_hand_eye`](@ref)'s.

The covariance is that of the staged estimate itself, first order in the
observation noise `ŝ²·diag(σ_rot², σ_trans²)` of the pose residual above
(`ŝ² = RSS/dof`): the rotation stages optimize different functionals than
that residual, so the estimate is not its stationary point and the residual's
Gauss–Newton `ŝ²(JᵀJ)⁻¹` would understate the scatter; instead the implicit
function theorem on the stages' own stationarity conditions (Kabsch, chordal
means, translation normal equations) carries the noise through. It is
reported with the observability diagnostics (`min_eigval`/`min_eigvec` of the
residual's `JᵀJ`), and is `Inf` when `JᵀJ` is rank-deficient.

# Units
The poses are plain or physical, all one form. Physical poses (translations
in lengths) are read in metres at entry — exactly, the stored value — the
solve runs on plain metres, and the calibration comes back physical (`Y`, `X`
in metres), equal after `ustrip(m, ·)` to the plain fit's on the poses'
`ustrip(m, ·)`. A supplied `σ_trans` is then a length too (a plain number
beside physical poses is a `DimensionError`). The metadata stay plain numbers
whose length is the translation's unit — metres for physical poses: the
residuals, the covariance's translation entries, `σ_trans`. Plain poses give
a plain calibration in their own unit, as before.

Inputs must be finite valid rigid transforms (`ArgumentError` otherwise);
domain-specific gross-outlier gates (depth windows, cheirality) belong at
the call site, upstream of this fit.

See also [`find_hand_eye`](@ref) for the ratio form `A_ij X = X B_ij` (no
pose prior, RANSAC) which eliminates `Y` instead of estimating it.
"""
function fit_robot_world_hand_eye(
        markers::AbstractDict{K, <:AbstractVector{<:Cspond{<:PlainEuclideanMap{3}, <:PlainEuclideanMap{3}}}};
        σ_rot::Union{Nothing, Real} = nothing,
        σ_trans::Union{Nothing, Real} = nothing,
        min_rel_angle::Real = deg2rad(_dflt("robot_world_hand_eye", "min_rel_angle_deg"))
    ) where {K}
    ids = block_order(markers)
    M = length(ids)
    n = sum(id -> length(markers[id]), ids; init = 0)
    n >= 4 || throw(ArgumentError("need at least 4 observations, got $n"))

    # One observation per correspondence, flattened marker by marker in `ids`
    # order. Each observation carries its own rig pose, so the init and the
    # residual index their A by the observation k.
    RA = RotMatrix{3, Float64, 9}[]; tA = SVector{3, Float64}[]
    RB = RotMatrix{3, Float64, 9}[]; tB = SVector{3, Float64}[]
    obs_marker = Int[]
    counts = _flatten_rig_observations(markers, ids, "observations") do m, id, j, RAk, tAk, b
        RBk, tBk = _finite_pose(b, id, j, "marker pose")
        push!(RA, RotMatrix{3, Float64}(RAk)); push!(tA, tAk)
        push!(RB, RotMatrix{3, Float64}(RBk)); push!(tB, tBk)
        push!(obs_marker, m)
        return 1
    end
    6n > 6 * (M + 1) || throw(
        ArgumentError(
            "underdetermined: $(n) observations for $(M + 1) transforms"
        )
    )
    by_marker = _obs_by_marker(obs_marker, M)

    # ---- staged linear solve ----------------------------------------------------
    RY, tY, RX, tX, pairs, axes_A = _robot_world_hand_eye_init(
        RA, tA, obs_marker, RB, tB, by_marker;
        min_rel_angle = Float64(min_rel_angle)
    )
    cal = RobotWorldHandEyeCalibration(
        EuclideanMap(RY, tY; check = false),
        Dict(ids[m] => EuclideanMap(RX[m], tX[m]; check = false) for m in 1:M)
    )
    obs = (; RA, tA, obs_marker, RB, tB, pairs, axes_A, by_marker)
    return _fit_robot_world_hand_eye_report(cal, _robot_world_hand_eye_manifold(cal, Val(M + 1)), obs, counts, σ_rot, σ_trans)
end

# Physical poses: stripped in metres once (with σ_trans), the plain fit, the
# calibration read back in metres once.
function fit_robot_world_hand_eye(
        markers::AbstractDict{K, <:AbstractVector{<:Cspond{<:_PhysicalPose3, <:_PhysicalPose3}}};
        σ_trans = nothing, kwargs...
    ) where {K}
    plain = Dict(id => _map_csponds(_plain_metres, _plain_metres, cs) for (id, cs) in markers)::
        AbstractDict{K, <:AbstractVector{<:Cspond{<:PlainEuclideanMap{3}, <:PlainEuclideanMap{3}}}}
    return _in_metres(fit_robot_world_hand_eye(plain; σ_trans = _metres(σ_trans), kwargs...))
end

# The report at the staged estimate, in its chart `P` (a function barrier: P's
# type carries the M + 1 blocks): noise-scale profiling, the diagnostic
# residual and its Jacobian, and the stage-condition covariance.
function _fit_robot_world_hand_eye_report(cal::RobotWorldHandEyeCalibration, P::ProductManifold, obs, counts, σ_rot, σ_trans)
    p = tangent_size(P)
    n = length(obs.obs_marker)
    r0 = _robot_world_hand_eye_residuals(zeros(p), P, obs, 1.0, 1.0)
    σr = σ_rot === nothing ? _block_rms(r0, 0) : Float64(σ_rot)
    σt = σ_trans === nothing ? _block_rms(r0, 3) : Float64(σ_trans)
    (σr > 0 && σt > 0) || throw(ArgumentError("profiled/provided σ must be positive"))

    r = _robot_world_hand_eye_residuals(zeros(p), P, obs, 1 / σr, 1 / σt)
    J = ForwardDiff.jacobian(z -> _robot_world_hand_eye_residuals(z, P, obs, 1 / σr, 1 / σt), zeros(p))
    # The staged estimate's covariance, first order (see the header): VGC's
    # square-system form F_θ⁻¹·F_ε·Σ_ε·F_εᵀ·F_θ⁻ᵀ, one 6×6 Σ_ε = ŝ²·diag(σr², σt²)
    # per observation.
    function covariance(s2)
        Fθ = ForwardDiff.jacobian(z -> _robot_world_hand_eye_stage_conditions(z, zeros(6n), P, obs), zeros(p))
        Fε = ForwardDiff.jacobian(z -> _robot_world_hand_eye_stage_conditions(zeros(p), z, P, obs), zeros(6n))
        Σε = SMatrix{6, 6, Float64, 36}(LinearAlgebra.Diagonal(s2 .* SVector(σr^2, σr^2, σr^2, σt^2, σt^2, σt^2)))
        return implicit_covariance(Fθ, Fε; Σx = fill(Σε, n))
    end
    return _robot_world_hand_eye_result(
        cal, r, J, 6, counts, covariance, σr, σt; retcode = :solved, iterations = 0, rms_rows = length(r)
    )
end

_block_rms(r, offset) = begin
    n6 = length(r) ÷ 6
    s = 0.0
    @inbounds for k in 1:n6, j in 1:3
        s += abs2(r[6 * (k - 1) + offset + j])
    end
    sqrt(max(s / (3 * n6), eps()))
end
