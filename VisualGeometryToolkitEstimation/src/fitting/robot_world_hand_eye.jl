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
# NLLS refinement (an LM on the stacked MRP ⊕ t increment) was implemented and
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
    _attach_metres(RobotWorldHandEyeCalibration(_plain_metres(Y), Dict{K, Rigid3}(id => convert(Rigid3, unlabel(_plain_metres(T))) for (id, T) in X)))

# A calibration's two forms, through its poses: the plain one in metres, and a
# plain one read as metres (the boundary of `fit_`/`refine_robot_world_hand_eye`).
_plain_metres(cal::RobotWorldHandEyeCalibration{K, <:_PhysicalPose3}) where {K} =
    RobotWorldHandEyeCalibration{K, Rigid3}(_plain_metres(cal.Y), Dict{K, Rigid3}(id => _plain_metres(T) for (id, T) in cal.X), cal.ids)
function _attach_metres(cal::RobotWorldHandEyeCalibration{K, Rigid3}) where {K}
    Y = _attach_metres(cal.Y)
    return RobotWorldHandEyeCalibration{K, typeof(Y)}(Y, Dict{K, typeof(Y)}(id => _attach_metres(T) for (id, T) in cal.X), cal.ids)
end

# The calibration's chart: VGC's product of `RigidExpManifold6` blocks
# [Y | X_{ids…}] anchored at it — the chart every covariance on the metadata is
# reported in. `Val(B)`, B = M + 1 blocks, keeps the product's type static.
_robot_world_hand_eye_manifold(cal::RobotWorldHandEyeCalibration, ::Val{B}) where {B} =
    ProductManifold(ntuple(b -> RigidExpManifold6(b == 1 ? cal.Y : cal.X[cal.ids[b - 1]]), Val(B))...)
# ... and back: the calibration from the poses (Y, X₁, …, X_M) in that order.
_robot_world_hand_eye_calibration(poses, ids::Vector{K}) where {K} =
    RobotWorldHandEyeCalibration(first(poses), Dict{K, Rigid3}(id => poses[m + 1] for (m, id) in enumerate(ids)))

"""
    RobotWorldHandEyeMetadata{K, S, L, W}

`FitResult` metadata of [`fit_robot_world_hand_eye`](@ref) (`S = Float64`) and of
[`refine_robot_world_hand_eye`](@ref) (`S = Nothing`: no pose-residual noise
scales; its `retcode_symbol(result)` is `:converged`, `:max_iterations`, `:max_time`, `:solve_failed` or `:stalled`) — one type, assembled in one place.
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
- `weights::W` — the per-datum weights (the fit: one per pose observation; the
  refinement: one per point), flattened marker by marker in `ids` order as
  `residuals` is: lazy unit weights (`unit_weights`) for a least-squares estimate
  (`L = Nothing`, `BinaryWeights`), the normalized loss weights `ρ′` of a
  robust one (`L` the strategy type, `IrlsWeights`). Read with `weights(result)`;
  `weight_kind(result)` dispatches on `L`.
- `scale::Float64`, `scale_source::ScaleSource` — the scale record of the
  robust-estimation contract (`noise_scale(result)`, `scale_source(result)`):
  the scale a robust estimate ran at and where it came from; `NaN` and
  `ScaleNone` for least squares or a width-only call.
- `loss_width::Float64` — the loss width the robust estimate ran at (`NaN`
  for least squares).
"""
struct RobotWorldHandEyeMetadata{K, S <: Union{Float64, Nothing}, L, W <: AbstractVector{Float64}}
    residuals::Dict{K, Vector{Float64}}
    s2::Float64
    dof::Int
    cov::Matrix{Float64}
    cov_Y::SMatrix{6, 6, Float64, 36}
    min_eigval::Float64
    min_eigvec::Vector{Float64}
    σ_rot::S
    σ_trans::S
    weights::W
    scale::Float64
    scale_source::VgtkCore.ScaleSource
    loss_width::Float64
end

# The weights' kind, read off the strategy type `L` (§12 of the robust-estimation
# contract, Core's trait of the strategy): least squares carries binary all-ones
# weights, Irls and Triggs the loss's weights.
VgtkCore.weight_kind(::Type{<:RobotWorldHandEyeMetadata{K, S, L}}) where {K, S, L} = VgtkCore.weight_kind(L)

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
# is `covariance(s2)` — the estimator's own, `nothing` when it has none. The
# robust record (`strategy`, the per-datum `weights`, the scale record and the
# loss width) defaults to least squares: `Nothing`, all-ones unit weights,
# `ScaleNone`, NaN. A robust estimate (`strategy !== nothing`) reports the
# weighted statistics of the robust-estimation contract: `s2` the weighted
# RSS over the weighted dof, `rms` the weighted root mean square (per datum as
# `rms_rows` makes it), and `n` the support (the data of positive weight);
# its `J` comes √w-scaled from the caller, so the eigen-diagnostics and the
# covariance are the weighted problem's.
function _robot_world_hand_eye_result(
        cal::RobotWorldHandEyeCalibration{K}, r::AbstractVector, J::AbstractMatrix, w::Int, counts, covariance,
        σ_rot::S, σ_trans::S; termination::Tuple{ReturnCode.T, StopRule}, iterations::Int, rms_rows::Int,
        strategy::L = nothing, weights::AbstractVector{Float64} = VgtkCore.unit_weights(length(r) ÷ w),
        scale::Float64 = NaN, scale_source::VgtkCore.ScaleSource = VgtkCore.ScaleNone, loss_width::Float64 = NaN
    ) where {K, S, L}
    N = size(J, 2)
    n = length(r) ÷ w
    length(weights) == n || throw(DimensionMismatch("$n data, $(length(weights)) weights"))
    dof, RSS, rms, support = if strategy === nothing
        length(r) - N, sum(abs2, r), sqrt(sum(abs2, r) / rms_rows), n
    else
        used, sum_w, rss_w = _weighted_support(weights, (sum(abs2, view(r, (w * (k - 1) + 1):(w * k))) for k in 1:n))
        dof_w = w * sum_w - N
        round(Int, dof_w), rss_w, sqrt(rss_w / (sum_w * rms_rows / n)), used
    end
    s2 = strategy === nothing ? RSS / dof : RSS / max(w * sum(weights) - N, eps())
    λ_min, v_min, cov = if all(isfinite, J)
        F = svd(J)                                    # singular values descending: JᵀJ's eigenpairs
        λ = F.S .^ 2
        C = λ[end] > _ROBOT_WORLD_HAND_EYE_RANK_RTOL * λ[1] ? covariance(s2) : nothing
        λ[end], Vector(F.V[:, end]), (C !== nothing && all(isfinite, C) ? C : fill(Inf, N, N))
    else
        NaN, fill(NaN, N), fill(Inf, N, N)
    end
    res_norms = split_blocks(point_norms!(Vector{Float64}(undef, n), r, n, w), cal.ids, counts)
    md = RobotWorldHandEyeMetadata{K, S, L, typeof(weights)}(
        res_norms, s2, dof, cov, SMatrix{6, 6, Float64, 36}(cov[1:6, 1:6]), λ_min, v_min, σ_rot, σ_trans,
        weights, scale, scale_source, loss_width
    )
    return FitResult(cal, termination..., iterations, rms, support, md)
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
        min_rel_angle::Float64,   # required: the caller holds the [robot_world_hand_eye] default
        wobs = nothing            # the robust fit's per-observation weights (nothing: least squares)
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
    R_Y = if wobs === nothing
        RotMatrix{3, Float64}(kabsch(as .=> bs)')                     # b = R_Yᵀ a  ⇒  Kabsch gives R_Yᵀ
    else                                                              # a pair weighs its two observations' weights
        pw = [wobs[k1] * wobs[k2] for (k1, k2) in pairs]
        count(>(0), pw) >= 2 || throw(ArgumentError(
            "fit_robot_world_hand_eye: the loss leaves fewer than 2 relative motions of positive weight; a " *
            "redescending loss needs a seed inside its width: use a monotone one (Irls{HuberShape}) or a wider `width`"))
        RotMatrix{3, Float64}(kabsch([pw[q] * as[q] => bs[q] for q in eachindex(pw) if pw[q] > 0])')
    end

    # 2. R_X per marker: R_B = R_Yᵀ R_A R_X  ⇒  R_X = R_Aᵀ R_Y R_B, chordal mean (weighted: Σ wₖRₖ projected)
    R_X = if wobs === nothing
        [chordal_mean([RotMatrix{3, Float64}(RA[k]' * R_Y * RB[k]) for k in by_marker[m]]) for m in 1:M]
    else
        map(1:M) do m
            ks = [k for k in by_marker[m] if wobs[k] > 0]
            isempty(ks) && throw(ArgumentError(
                "fit_robot_world_hand_eye: the loss gives every observation of marker $m (in sorted id order) zero weight " *
                "at the least-squares seed; a redescending loss needs a seed inside its width: use a monotone one " *
                "(Irls{HuberShape}), a wider `width`, or gate the gross poses first"))
            RotMatrix{3, Float64}(chordal_mean([wobs[k] * SMatrix{3, 3, Float64, 9}(RA[k]' * R_Y * RB[k]) for k in ks]))
        end
    end

    # 3. All translations jointly:  t_B = R_Yᵀ R_A t_X − R_Yᵀ t_Y + R_Yᵀ t_A
    #    unknown u = [t_Y ; t_X₁ … t_X_M] — exactly optimal given the rotations.
    #    Solved in a normalized space (the stack's normalize-LS rule): each side's
    #    translations about their centroid c_A, c_B, at ONE isotropic scale s — the
    #    rig side's HartleyNormalization (VisualGeometryToolkitCore, 3-D: mean distance
    #    √3) — as t′_A = s(t_A − c_A), t′_B = s(t_B − c_B). The rows keep their form,
    #    with t′_X = s·t_X and t′_Y = s(t_Y − c_A + R_Y c_B), unnormalized on the way out.
    #    The rotation blocks of the design are untouched (its singular values are
    #    the same); what moves is the right-hand side's cancellation when a frame
    #    sits far from its origin.
    n = length(obs_marker)
    NA = wobs === nothing ? AffineMap(HartleyNormalization(), tA) : AffineMap(HartleyNormalization(), tA, wobs)   # the observations' weights, when weighted
    NB = wobs === nothing ? AffineMap(HartleyNormalization(), tB) : AffineMap(HartleyNormalization(), tB, wobs)
    s = NA.linear[1, 1]
    cA = inv(NA)(zero(SVector{3, Float64})); cB = inv(NB)(zero(SVector{3, Float64}))
    Amat = zeros(3n, 3 * (M + 1))
    rhs = zeros(3n)
    RYt = SMatrix{3, 3, Float64, 9}(R_Y')
    for k in 1:n
        m = obs_marker[k]
        rows = (3k - 2):(3k)
        Amat[rows, 1:3] .= -RYt
        Amat[rows, (3m + 1):(3m + 3)] .= RYt * SMatrix{3, 3, Float64, 9}(RA[k])
        rhs[rows] .= s * (tB[k] - cB) - RYt * NA(tA[k])
        if wobs !== nothing                                            # the weighted least squares: rows × √wₖ
            sk = sqrt(wobs[k])
            Amat[rows, :] .*= sk
            rhs[rows] .*= sk
        end
    end
    u = Amat \ rhs
    t_Y = SVector{3, Float64}(u[1], u[2], u[3]) / s + cA - R_Y * cB
    t_X = [SVector{3, Float64}(u[3m + 1], u[3m + 2], u[3m + 3]) / s for m in 1:M]

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
    (; RA, tA, obs_marker, RB, tB, pairs, axes_A, by_marker, wobs) = st
    T = promote_type(eltype(θ), eltype(ε))
    M = length(by_marker)
    n = length(obs_marker)

    poses = retract(P, θ)
    R_Y = rotation(first(poses)); t_Y = translation(first(poses))
    Bε = [retract(RigidExpManifold6(RB[k], tB[k]), view(ε, (6k - 5):(6k))) for k in 1:n]
    RBε(k) = rotation(Bε[k])

    F = Vector{T}(undef, 6 * (M + 1))
    F[1:3] .= kabsch_stationarity(
        R_Y', [(_obs_weight(wobs, k1) * _obs_weight(wobs, k2)) * axes_A[p] => log_so3(SMatrix{3, 3}(RBε(k1) * RBε(k2)'))
               for (p, (k1, k2)) in enumerate(pairs)]
    )
    nY = zero(SVector{3, T})
    for m in 1:M
        X = poses[m + 1]
        R_X = rotation(X); t_X = translation(X)
        F[(3m + 1):(3m + 3)] .= chordal_mean_stationarity(R_X, [_obs_weight(wobs, k) * (RA[k]' * R_Y * RBε(k)) for k in by_marker[m]])
        nX = zero(SVector{3, T})
        for k in by_marker[m]
            R_A = RA[k]
            e = R_Y' * (R_A * t_X + tA[k] - t_Y) - translation(Bε[k])   # row block k of (A u − rhs)
            RYe = _obs_weight(wobs, k) * (R_Y * e)
            nY -= RYe
            nX += R_A' * RYe
        end
        F[(3 * (M + 1) + 3m + 1):(3 * (M + 1) + 3m + 3)] .= nX
    end
    F[(3 * (M + 1) + 1):(3 * (M + 1) + 3)] .= nY
    return F
end

# An observation's weight in the stage conditions: `true` (an exact ×1) for
# least squares, the robust fit's fixed weight otherwise.
_obs_weight(::Nothing, _k) = true
_obs_weight(w::AbstractVector, k) = w[k]

# -----------------------------------------------------------------------------
# fit_robot_world_hand_eye
# -----------------------------------------------------------------------------

"""
    fit_robot_world_hand_eye(markers; σ_rot=nothing, σ_trans=nothing, min_rel_angle,
                             optimization_strategy = nothing, scale | width, tol)
        -> FitResult{RobotWorldHandEyeCalibration{K}, RobotWorldHandEyeMetadata{K, Float64}}

`min_rel_angle` defaults to `configs/default.toml` `[robot_world_hand_eye]`
`min_rel_angle_deg` (5°), converted to radians.

Staged linear robot-world / multi-marker hand-eye calibration
`A_i X_m = Y B_i^m`: one shared `Y = T_WC` and per-marker `X_m = T_RPm`,
fit from per-(frame, marker) pose observations. No RANSAC: with a pose
prior (mocap) contamination is pre-gateable at the call site, or the fit
downweights it itself with a fixed-scale IRLS (`optimization_strategy`,
below).

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
- `optimization_strategy` — `nothing` (the default: the staged linear solve
  above, least squares), or an [`Irls`](@extref VisualGeometryToolkitCore.Irls)`{K}` strategy: the staged solve
  reweighted per observation at ONE loss width on the whitened 6-vector
  residual (codimension 6), iterated to `tol` or the strategy's `max_iter`.
  Each stage takes the weights: a Kabsch pair weighs `wₖ₁·wₖ₂`, the chordal
  means and the translation least squares `wₖ`. A gross observation (a pose a
  registration placed tens of degrees and metres wrong) then gets weight ≈ 0
  instead of tilting `Y` and every `X`. The robust fit REQUIRES `σ_rot` and
  `σ_trans`: they are the whitening, the units of its scale, and profiling them
  from the contaminated residuals would re-estimate a scale the fit holds
  fixed (`ArgumentError` without them). A Triggs strategy is refused (a
  refine's: [`refine_robot_world_hand_eye`](@ref)).
- `scale` | `width` — with an `Irls` strategy, EXACTLY ONE of: `scale = σ`
  (the whitened residual's noise scale per component, dimensionless: 1 when
  `σ_rot`/`σ_trans` are the true noise), `scale = result` (an upstream robust
  result's `noise_scale`), or `width = w` (the loss width on the whitened
  residual norm). The width is `w`, or `c · σ` with `c` the strategy's
  efficiency constant, by default the shape's at codimension 6
  (`efficiency_constant(K, 6)`: Huber 1.8693, Welsch 3.9646). Neither, both,
  or one given to the least-squares fit is an `ArgumentError`.
- `tol` — the robust fit's convergence: an iteration moves every transform by
  less than `tol` in the whitened units (rotation angle / `σ_rot`, translation
  / `σ_trans`); default configs/default.toml `[robot_world_hand_eye] tol`.

The covariance is that of the staged estimate itself, first order in the
observation noise `ŝ²·diag(σ_rot², σ_trans²)` of the pose residual above
(`ŝ² = RSS/dof`): the rotation stages optimize different functionals than
that residual, so the estimate is not its stationary point and the residual's
Gauss–Newton `ŝ²(JᵀJ)⁻¹` would understate the scatter; instead the implicit
function theorem on the stages' own stationarity conditions (Kabsch, chordal
means, translation normal equations) carries the noise through. It is
reported with the observability diagnostics (`min_eigval`/`min_eigvec` of the
residual's `JᵀJ`), and is `Inf` when `JᵀJ` is rank-deficient. A robust fit
reports the same at its final weights, held fixed: the weighted stage
conditions, `√w`-scaled residual rows, `ŝ²` the weighted RSS over the weighted
dof; `rms` is weighted and `n` the support (observations of positive weight).
The metadata's `weights` (one per observation, marker by marker in `ids`
order), `scale`, `scale_source` and `loss_width` are the robust-estimation
contract's record; least squares carries unit weights and `ScaleNone`.

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

Inputs must be finite valid rigid transforms (`ArgumentError` otherwise).
Domain-specific gates (depth windows, cheirality) still belong at the call
site; a robust `optimization_strategy` handles the gross poses no gate
catches (a sparse, badly registered frame).

See also [`find_hand_eye`](@ref) for the ratio form `A_ij X = X B_ij` (no
pose prior, RANSAC) which eliminates `Y` instead of estimating it.
"""
function fit_robot_world_hand_eye(
        markers::AbstractDict{K, <:AbstractVector{<:Cspond{<:PlainEuclideanMap{3}, <:PlainEuclideanMap{3}}}};
        σ_rot::Union{Nothing, Real} = nothing,
        σ_trans::Union{Nothing, Real} = nothing,
        min_rel_angle::Real = deg2rad(_dflt("robot_world_hand_eye", "min_rel_angle_deg")),
        optimization_strategy::Union{Nothing, AbstractOptimizationStrategy} = nothing,
        local_optimization = _RetiredKeyword(),   # refused, naming its successor
        scale = nothing, width::Union{Nothing, Real} = nothing,
        tol::Real = _dflt("robot_world_hand_eye", "tol"),
    ) where {K}
    _refuse_retired_keyword(local_optimization, :local_optimization, :optimization_strategy)
    f = _fit_irls("fit_robot_world_hand_eye", optimization_strategy, scale, width, 6)
    f === nothing || (σ_rot !== nothing && σ_trans !== nothing) || throw(ArgumentError(
        "fit_robot_world_hand_eye: a robust fit needs `σ_rot` and `σ_trans` — the whitening its scale is in; " *
        "profiling them from contaminated residuals would re-estimate the scale the fit holds fixed"))
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
    obs = (; RA, tA, obs_marker, RB, tB, pairs, axes_A, by_marker, wobs = nothing)
    f === nothing && return _fit_robot_world_hand_eye_report(
        cal, _robot_world_hand_eye_manifold(cal, Val(M + 1)), obs, counts, σ_rot, σ_trans)

    # ---- the fixed-scale IRLS: weights at the current estimate, the weighted staged solve
    σr, σt = Float64(σ_rot), Float64(σ_trans)
    (σr > 0 && σt > 0) || throw(ArgumentError("provided σ must be positive"))
    wobs = ones(n)
    tol_met = false; iter = 0
    while iter < _fit_max_iter(optimization_strategy)
        iter += 1
        _robot_world_hand_eye_weights!(wobs, f, cal, Val(M + 1), obs, σr, σt)
        RY, tY, RX, tX, pairs, axes_A = _robot_world_hand_eye_init(
            RA, tA, obs_marker, RB, tB, by_marker; min_rel_angle = Float64(min_rel_angle), wobs
        )
        new = RobotWorldHandEyeCalibration(
            EuclideanMap(RY, tY; check = false),
            Dict(ids[m] => EuclideanMap(RX[m], tX[m]; check = false) for m in 1:M)
        )
        Δ = _robot_world_hand_eye_move(new, cal, σr, σt)
        cal = new
        obs = (; RA, tA, obs_marker, RB, tB, pairs, axes_A, by_marker, wobs = nothing)
        Δ < tol && (tol_met = true; break)
    end
    return _fit_robot_world_hand_eye_report(
        cal, _robot_world_hand_eye_manifold(cal, Val(M + 1)), obs, counts, σ_rot, σ_trans;
        f, strategy = optimization_strategy, termination = _irls_record(tol_met, false), iterations = iter
    )
end

# The robust fit's weights at `cal`: the loss weight of each observation's
# whitened 6-residual norm (a function barrier: the chart's type carries B blocks).
function _robot_world_hand_eye_weights!(wobs, f, cal, ::Val{B}, obs, σr, σt) where {B}
    P = _robot_world_hand_eye_manifold(cal, Val(B))
    r = _robot_world_hand_eye_residuals(zeros(tangent_size(P)), P, obs, 1 / σr, 1 / σt)
    n = length(wobs)
    return _irls_weights!(wobs, f, point_norms!(Vector{Float64}(undef, n), r, n, 6))
end

# The largest move of any transform between two calibrations, in the whitened
# units: rotation angle / σ_rot, translation / σ_trans.
function _robot_world_hand_eye_move(a::RobotWorldHandEyeCalibration, b::RobotWorldHandEyeCalibration, σr, σt)
    d(T, U) = max(Rotations.rotation_angle(RotMatrix{3, Float64}(rotation(T) * rotation(U)')) / σr,
                  norm(translation(T) - translation(U)) / σt)
    return max(d(a.Y, b.Y), maximum(id -> d(a.X[id], b.X[id]), a.ids))
end

# Physical poses: stripped in metres once (with σ_trans), the plain fit, the
# calibration read back in metres once.
function fit_robot_world_hand_eye(
        markers::AbstractDict{K, <:AbstractVector{<:Cspond{<:_PhysicalPose3, <:_PhysicalPose3}}};
        σ_trans = nothing, kwargs...
    ) where {K}
    plain = Dict(id => _map_csponds(_plain_metres, _plain_metres, cs) for (id, cs) in markers)::
        AbstractDict{K, <:AbstractVector{<:Cspond{<:PlainEuclideanMap{3}, <:PlainEuclideanMap{3}}}}
    return _attach_metres(fit_robot_world_hand_eye(plain; σ_trans = _metres(σ_trans), kwargs...))
end

# The report at the staged estimate, in its chart `P` (a function barrier: P's
# type carries the M + 1 blocks): noise-scale profiling, the diagnostic
# residual and its Jacobian, and the stage-condition covariance.
function _fit_robot_world_hand_eye_report(cal::RobotWorldHandEyeCalibration, P::ProductManifold, obs, counts, σ_rot, σ_trans;
                                          f = nothing, strategy = nothing, termination::Tuple{ReturnCode.T, StopRule} = (ReturnCode.Success, ClosedForm), iterations::Int = 0)
    p = tangent_size(P)
    n = length(obs.obs_marker)
    r0 = _robot_world_hand_eye_residuals(zeros(p), P, obs, 1.0, 1.0)
    σr = σ_rot === nothing ? _block_rms(r0, 0) : Float64(σ_rot)
    σt = σ_trans === nothing ? _block_rms(r0, 3) : Float64(σ_trans)
    (σr > 0 && σt > 0) || throw(ArgumentError("profiled/provided σ must be positive"))

    r = _robot_world_hand_eye_residuals(zeros(p), P, obs, 1 / σr, 1 / σt)
    J = ForwardDiff.jacobian(z -> _robot_world_hand_eye_residuals(z, P, obs, 1 / σr, 1 / σt), zeros(p))
    wts = _fit_weights_storage(f, n)
    if f !== nothing                                   # the final weights, held fixed in the report
        _irls_weights!(wts, f, point_norms!(Vector{Float64}(undef, n), r, n, 6))
        obs = merge(obs, (; wobs = wts))
        J = repeat(sqrt.(wts); inner = 6) .* J
    end
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
        cal, r, J, 6, counts, covariance, σr, σt; termination, iterations, rms_rows = length(r),
        strategy, weights = wts, (_robot_world_hand_eye_scale_record(f))...
    )
end

_robot_world_hand_eye_scale_record(f) = NamedTuple{(:scale, :scale_source, :loss_width)}(_fit_scale_record(f))

_block_rms(r, offset) = begin
    n6 = length(r) ÷ 6
    s = 0.0
    @inbounds for k in 1:n6, j in 1:3
        s += abs2(r[6 * (k - 1) + offset + j])
    end
    sqrt(max(s / (3 * n6), eps()))
end
