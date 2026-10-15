# =============================================================================
# refine_robot_world_hand_eye — reprojection-error refinement of a robot-world calibration
# =============================================================================
#
# The measurement-level upgrade `fit_robot_world_hand_eye`'s header names: refine the
# M + 1 transforms of A_i X_m = Y B_i^m against the FEATURES — the marker
# points' reprojection error — rather than against per-frame PnP poses. With
# the camera and the rig poses A_i fixed, marker m's point X seen in frame i
# lands at
#
#     û = project(camera, Y⁻¹ ∘ A_i ∘ X_m (X)),
#
# and the estimate minimizes Σ‖û − u‖² over [Y | X₁ … X_M] in the calibration's
# chart — `fit_robot_world_hand_eye`'s: VGC's `ProductManifold` of M + 1
# `RigidExpManifold6` blocks (exp(δω)·R, t + δt), anchored at the guess
# (`_robot_world_hand_eye_manifold`) — with VGC's chart LM (`lmsolve_chart!`:
# FastLevenbergMarquardt's `lmsolve!`, N = 6(M + 1)), re-anchored at the
# solution by the product's `recenter`. A
# point depends on Y and its own X_m only, so its 2 × 12 Jacobian is
# differentiated (ForwardDiff, through the retraction of the product of those
# two blocks) and scattered into the 2n × N buffer.
#
# The covariance is the stack's one model covariance, VGC's
# `implicit_covariance`, over the points' reprojection rows at the estimate
# (block form; unit isotropic pixel noise, so (JᵀJ)⁻¹), scaled by
# ŝ² = RSS/dof; the result is assembled by `fit_robot_world_hand_eye`'s one assembly
# (`_robot_world_hand_eye_result`): the joint `cov` in block order [Y | X₁ … X_M], the
# marginal `cov_Y`, the smallest eigenpair of JᵀJ and the rank rule. Unlike
# `fit_robot_world_hand_eye`'s, this one is the covariance of the estimator's own
# stationary point (an MLE under isotropic Gaussian pixel noise).
# =============================================================================

# The chart LM is static in its N = 6(M + 1) unknowns — the iterate is an
# `SVector{N}` and the product chart unrolls over its M + 1 blocks, compiled per
# M; its normal equations are FastLevenbergMarquardt's LAPACK Cholesky above
# N = 12 (VGC `_lm_solver`), so no marker count is capped. (Until 2026-10-01
# the solve was a static N × N Cholesky too, whose compile cost capped M at 4.)

"""
    refine_robot_world_hand_eye(guess::RobotWorldHandEyeCalibration{K}, camera::CameraModel, markers;
                       max_iter)   # default: configs/default.toml [refine_robot_world_hand_eye]
        -> FitResult{RobotWorldHandEyeCalibration{K}, RobotWorldHandEyeMetadata{K, Nothing}}

Refine a robot-world calibration `A_i X_m = Y B_i^m` against the reprojection
error of the markers' FEATURES: minimize `Σ‖project(camera, Y⁻¹ ∘ A_i ∘ X_m (X)) − u‖²`
over the shared `Y = T_WC` and the per-marker `X_m = T_RPm`, starting at
`guess` (e.g. [`fit_robot_world_hand_eye`](@ref)'s estimate). The camera and the rig
poses `A` are FIXED; nothing is re-estimated from scratch and the family is
unchanged — the `refine` contract.

Robot-world/hand-eye calibration is the literature's name for `AX = YB`
(Zhuang, Roth & Shah 1994; Dornaika & Horaud 1998). Ours is the EYE-TO-HAND
configuration — the camera fixed in the world, the targets on the moving rig —
with frames in the stack's `T_AB` notation: `Y = T_WC`, `X_m = T_RPm`,
`A_i = T_WR` (the rig pose at frame i) and `B_i^m = T_CPm` (marker m in the
camera at frame i). Function names do not encode frames.

# Arguments
- `guess::RobotWorldHandEyeCalibration{K}` — the starting `Y` and the keyed `X`
- `camera::CameraModel` — the (fixed) camera; `project(camera, X_C)` maps a
  camera-frame point to pixels
- `markers::AbstractDict{K, <:AbstractVector{<:Cspond}}` — per marker id, one
  element per frame it is seen in: that frame's rig pose `A_i = T_WR`
  (`unlabel`ed at the boundary) paired with the features seen with it,
  `A_i => [X => u, …]` — `X` a point in the marker's frame, `u` its measured
  pixel. The key sets must match, `keys(markers) == Set(guess.ids)`, and every
  marker must have at least one feature (`ArgumentError` otherwise): every
  marker of the guess is a parameter of this refinement and is refined.

# Keywords
- `max_iter` — the LM's iteration cap.

Any number of markers: the LM is static in its 6(M + 1) unknowns and compiled
once per marker count. Plain least squares — the observations must be inliers (gate them
upstream, as the per-board detection does). Rig poses, points and pixels must
be finite, and every point must project at the guess (in front of the camera,
inside the lens's domain): an `ArgumentError` names the first that does not.

The covariance is VGC's `implicit_covariance` over the reprojection rows at
the estimate times `ŝ² = RSS/dof` — see [`RobotWorldHandEyeMetadata`](@ref).
"""
function refine_robot_world_hand_eye(
        guess::RobotWorldHandEyeCalibration{K}, camera::CameraModel,
        markers::AbstractDict{
            K, <:AbstractVector{<:Cspond{<:EuclideanMap{3}, <:AbstractVector{<:Cspond{<:StaticVector{3}, <:StaticVector{2}}}}},
        };
        max_iter::Int = _dflt("refine_robot_world_hand_eye", "max_iter")
    ) where {K}
    ids = block_order(markers, guess.X)             # the key sets must match: every marker of the guess is refined
    M = length(ids)
    # Flattened marker by marker in `ids` order: each element's rig pose once
    # (RA/tA, indexed by pt_frame), each of its features one point.
    RA = SMatrix{3, 3, Float64, 9}[]; tA = SVector{3, Float64}[]
    pt_frame = Int[]; pt_marker = Int[]
    Xs = SVector{3, Float64}[]; us = SVector{2, Float64}[]
    counts = _flatten_rig_observations(markers, ids, "features") do m, id, j, RAj, tAj, features
        push!(RA, RAj); push!(tA, tAj)
        f = length(RA)
        for (q, pc) in enumerate(features)
            push!(pt_frame, f); push!(pt_marker, m)
            push!(Xs, SVector{3, Float64}(first(pc)[1], first(pc)[2], first(pc)[3]))
            push!(us, SVector{2, Float64}(last(pc)[1], last(pc)[2]))
            (all(isfinite, Xs[end]) && all(isfinite, us[end])) ||
                throw(ArgumentError("marker $id, element $j, feature $q: non-finite point or pixel"))
        end
        return length(features)
    end
    n = length(us)
    N = 6 * (M + 1)
    2n > N || throw(ArgumentError("underdetermined: $n points ($(2n) residuals) for $N unknowns"))
    return _refine_robot_world_hand_eye(Val(M + 1), guess, counts, camera, RA, tA, pt_frame, pt_marker, Xs, us, max_iter)
end

# One point's reprojection residual as a function of its 12 chart coordinates
# (δY, δX_m) in the product of its two blocks' charts `chY`, `chX` —
# eltype-generic, so ForwardDiff Duals flow through the retraction and the lens.
@inline function _rw_point_residual(z::SVector{12}, chY, chX, RAi, tAi, X, u, camera)
    T = eltype(z)
    Y, Xm = retract(ProductManifold(chY, chX), z)
    Pw = RAi * Xm(X) + tAi                                      # marker → rig → world
    return SVector{2, T}(VisualGeometryCore.reprojection_error(camera, inv(Y), Pw, u))   # world → camera → pixel
end

# The linearization state of one robot-world refinement call: the calibration's
# chart `P` (`_robot_world_hand_eye_manifold`), the arm poses, the frame and marker index
# of every observation, the board points, their pixels and the camera.
struct _RobotWorldHandEyeLm{TP, TR, TT, TF, TM, TX, TU, TC}
    P::TP
    RA::TR
    tA::TT
    fr::TF
    mk::TM
    Xs::TX
    us::TU
    camera::TC
end

# Point k's residual at the stacked chart coordinates `x` — through its 12
# coordinates (δY, δX_m), the parts of `x` it depends on.
@inline function _rw_point_residual(lm::_RobotWorldHandEyeLm, parts, k, z = vcat(parts[1], parts[lm.mk[k] + 1]))
    chs = lm.P.charts
    f = lm.fr[k]
    return _rw_point_residual(z, chs[1], chs[lm.mk[k] + 1], lm.RA[f], lm.tA[f], lm.Xs[k], lm.us[k], lm.camera)
end

# The reprojection rows in the stacked chart at `x` (`residual_rows!`).
function residual_rows!(f, lm::_RobotWorldHandEyeLm, x::SVector)
    parts = manifold_parts(lm.P, x)
    @inbounds for k in eachindex(lm.us)
        r = _rw_point_residual(lm, parts, k)
        f[2k - 1] = r[1]; f[2k] = r[2]
    end
    return f
end

# The Jacobian by per-point BLOCKS, not VGC's dense default (ForwardDiff through
# `residual_rows!`): a point depends on 12 of the N = 6(M + 1) coordinates, so
# one 12-partial sweep per point replaces ⌈N/chunk⌉ sweeps over all points.
# Measured 2026-09-30 against the dense default (per Jacobian): M = 1, n = 300:
# 0.12 vs 0.36 ms (3.0×); M = 3, n = 900: 0.32 vs 0.68 ms (2.1×); M = 4,
# n = 1200: 0.49 vs 1.35 ms (2.8×); the two agree to 3.4e-13.
function residual_jacobian_rows!(J, lm::_RobotWorldHandEyeLm, x::SVector)
    fill!(J, 0.0)
    parts = manifold_parts(lm.P, x)
    @inbounds for k in eachindex(lm.us)
        m = lm.mk[k]
        Jk = ForwardDiff.jacobian(z -> _rw_point_residual(lm, parts, k, z), vcat(parts[1], parts[m + 1]))
        for a in 1:2, j in 1:6
            J[2k - 2 + a, j] = Jk[a, j]
            J[2k - 2 + a, 6m + j] = Jk[a, 6 + j]
        end
    end
    return J
end

# Static in the B = M + 1 blocks of the chart (a function barrier).
function _refine_robot_world_hand_eye(
        ::Val{B}, guess::RobotWorldHandEyeCalibration, counts, camera, RA, tA, fr, mk, Xs, us, max_iter
    ) where {B}
    n = length(us)
    N = 6B
    lm = _RobotWorldHandEyeLm(_robot_world_hand_eye_manifold(guess, Val(B)), RA, tA, fr, mk, Xs, us, camera)
    # Every point must project at the guess: NaN is the lens's answer for a
    # ray it does not image (behind the camera, outside its domain), and one
    # NaN row would silently stall the LM.
    r0 = residual_rows!(Vector{Float64}(undef, 2n), lm, zero(SVector{N, Float64}))
    for k in 1:n
        (isfinite(r0[2k - 1]) && isfinite(r0[2k])) || throw(
            ArgumentError(
                "marker $(guess.ids[mk[k]]): point $(Xs[k]) does not project at the guess " *
                    "(behind the camera or outside the lens's domain)"
            )
        )
    end

    ews = _lm_workspace(nothing, 2n, Val(N))       # the one LO-buffer reuse policy (interface.jl)
    x, _, conv, iterations = lmsolve_chart!(ews, lm; maxit = max_iter, stationarity_tol = _CHART_LM_STATIONARITY_TOL)
    all(isfinite, x) || throw(ErrorException("refine_robot_world_hand_eye: the LM step is not finite"))

    # The estimate: the chart re-anchored at the solution (the covariance's chart).
    lm_hat = _RobotWorldHandEyeLm(recenter(lm.P, x), RA, tA, fr, mk, Xs, us, camera)
    x0 = zero(SVector{N, Float64})
    r = residual_rows!(Vector{Float64}(undef, 2n), lm_hat, x0)
    J = residual_jacobian_rows!(Matrix{Float64}(undef, 2n, N), lm_hat, x0)
    # The covariance's blocks are the ELEMENTS — one rig pose's features, whose
    # points are contiguous (flattened element by element) — each a dense view
    # of its rows of J with measurement Jacobian −I: the same (JᵀJ)⁻¹ as per-
    # point blocks, without a static 2 × N block and an N × N static product
    # per point. Measured 2026-10-01 at M = 20 (N = 126, 6000 points): the
    # per-point form took 370 ms and 2.2 GB per call, 4× the LM itself.
    stops = [k for k in 1:n if k == n || fr[k + 1] != fr[k]]
    starts = [1; stops[1:(end - 1)] .+ 1]
    rows(e) = (2starts[e] - 1):(2stops[e])
    function covariance(s2)
        Σ̃ = implicit_covariance(e -> (view(J, rows(e), :), Diagonal(fill(-1.0, length(rows(e))))), eachindex(starts))
        return Σ̃ === nothing ? nothing : s2 .* Σ̃
    end
    return _robot_world_hand_eye_result(
        _robot_world_hand_eye_calibration(lm_hat.P, guess.ids), r, J, 2, counts, covariance, nothing, nothing;
        retcode = VisualGeometryCore.lm_retcode(conv), iterations, rms_rows = n
    )
end
